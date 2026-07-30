"""On-demand camera acquisition and synthetic-image generation."""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
from html import escape
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .dataset import (
    PipelineError,
    read_json,
    sha256,
    utc_now,
    validate_dataset_design,
    write_json,
)


USER_AGENT = "mai-research/0.4 (+https://github.com/athen-lab/mai)"
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
JPEG_SIGNATURE = b"\xff\xd8\xff"
REJECTED_EDITORS = (
    "adobe",
    "photoshop",
    "lightroom",
    "gimp",
    "capture one",
    "affinity",
    "snapseed",
)
SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "arranged",
    "at",
    "beside",
    "between",
    "by",
    "clinging",
    "covering",
    "cut",
    "displayed",
    "displaying",
    "divided",
    "falling",
    "filled",
    "floating",
    "from",
    "grazing",
    "growing",
    "hanging",
    "in",
    "into",
    "leaning",
    "of",
    "on",
    "over",
    "parked",
    "pulled",
    "reflecting",
    "resting",
    "rising",
    "sitting",
    "standing",
    "surrounded",
    "swimming",
    "the",
    "through",
    "topped",
    "traveling",
    "under",
    "viewed",
    "waiting",
    "walking",
    "winding",
    "with",
}
QUERY_SCAFFOLDING = {"a", "an", "of", "the"}
DEFAULT_REJECTION_REASONS = {
    "border_or_collage",
    "corrupt_or_blank",
    "literal_prompt_scaffolding",
    "non_photographic_style",
    "semantic_mismatch",
    "watermark",
}

CameraFetcher = Callable[
    [dict[str, Any], dict[str, Any], Path],
    dict[str, Any],
]
GeneratorRunner = Callable[
    [dict[str, Any], dict[str, Any], int | None, Path],
    dict[str, Any],
]
RealSourceLoader = Callable[
    [dict[str, Any], int],
    list[dict[str, Any]],
]
CaptionRunner = Callable[
    [Path, dict[str, Any]],
    dict[str, Any] | str,
]
QARunner = Callable[
    [Path, str | None, dict[str, Any]],
    dict[str, Any],
]

LICENSE_ALIASES = {
    "cc-by-4.0": "CC-BY-4.0",
    "cc by 4.0": "CC-BY-4.0",
    "https://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "cc0-1.0": "CC0-1.0",
    "cc0 1.0": "CC0-1.0",
    "https://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
}
LICENSE_URLS = {
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
}
CAPTION_SCAFFOLDING = (
    r"^\s*(?:this|the)\s+(?:image|photo|photograph|picture)\s+"
    r"(?:shows|depicts|features|captures|contains|is\s+of)\s+",
    r"^\s*in\s+(?:this|the)\s+(?:image|photo|photograph|picture)\s*,?\s*",
    r"^\s*(?:we|you)\s+can\s+see\s+",
)
CAPTION_AESTHETIC_WORDS = {
    "beautiful",
    "breathtaking",
    "captivating",
    "charming",
    "gorgeous",
    "lovely",
    "picturesque",
    "stunning",
}

_CAPTION_MODEL: dict[str, Any] = {}
_CLIP_MODEL: dict[str, Any] = {}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def normalize_caption(raw_caption: str, policy: str = "content-only-v1") -> str:
    """Normalize a model caption into a factual, generator-facing description."""
    if policy != "content-only-v1":
        raise PipelineError(f"unknown caption normalization policy: {policy}")
    if not isinstance(raw_caption, str) or not raw_caption.strip():
        raise PipelineError("captioner returned an empty caption")
    text = " ".join(raw_caption.replace("\n", " ").split()).strip(" \"'")
    for pattern in CAPTION_SCAFFOLDING:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:likely|probably|possibly|apparently)\b\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:appears|seems)\s+to\s+(?:show|depict|be)\b\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:professionally\s+shot|high[- ]resolution|wide[- ]angle|"
        r"telephoto|cinematic|award[- ]winning)\b\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    aesthetic = "|".join(sorted(CAPTION_AESTHETIC_WORDS))
    text = re.sub(
        rf"\b(?:{aesthetic})\b[\s,]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;:-")
    if not text:
        raise PipelineError(
            "caption normalization removed all factual content"
        )
    return text[0].upper() + text[1:] + "."


def _render_caption_prompt(
    group_id: str,
    normalized_caption: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    template = str(policy["template"])
    caption = normalized_caption.rstrip(".")
    try:
        text = template.format(caption=caption).strip()
    except (KeyError, ValueError) as error:
        raise PipelineError(f"invalid caption prompt template: {error}") from error
    if not text:
        raise PipelineError("caption prompt template rendered an empty prompt")
    return {
        "prompt_id": f"caption-{group_id}",
        "text": text,
        "frozen": True,
    }


def _canonical_license(
    raw_value: Any,
    policy: dict[str, Any],
) -> tuple[str, str] | None:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    aliases = {
        **LICENSE_ALIASES,
        **{
            str(key).casefold(): str(value)
            for key, value in policy.get("aliases", {}).items()
        },
    }
    raw = raw_value.strip()
    canonical = aliases.get(raw.casefold(), raw)
    allowed = policy.get("allowed", [])
    if canonical not in allowed:
        return None
    return canonical, LICENSE_URLS.get(canonical, raw)


def load_huggingface_rows(
    config: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    """Load a deterministic, bounded row stream from a pinned HF revision."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise PipelineError(
            "Hugging Face source dependencies are missing; install with "
            "`python3 -m pip install -e '.[source]'`"
        ) from error
    try:
        rows = load_dataset(
            config["dataset_id"],
            revision=config["revision"],
            split=config["split"],
            streaming=True,
        )
        rows = rows.shuffle(
            seed=int(config["sample_seed"]),
            buffer_size=int(config.get("shuffle_buffer_size", 10_000)),
        )
        return [dict(row) for row in islice(rows, limit)]
    except Exception as error:
        raise PipelineError(
            f"cannot stream {config['dataset_id']}@{config['revision']}: "
            f"{error}"
        ) from error


def _materialize_hf_image(value: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        if value.startswith(("https://", "http://")):
            _download(value, target)
            return
        source = Path(value).expanduser().resolve()
        if not source.is_file():
            raise PipelineError(f"HF image path does not exist: {source}")
        shutil.copyfile(source, target)
        return
    if isinstance(value, dict):
        payload = value.get("bytes")
        if isinstance(payload, bytes):
            target.write_bytes(payload)
            return
        path = value.get("path")
        if isinstance(path, str):
            _materialize_hf_image(path, target)
            return
    if hasattr(value, "save"):
        image_format = getattr(value, "format", None) or "PNG"
        value.save(target, format=image_format)
        return
    raise PipelineError(
        f"unsupported HF image value: {type(value).__name__}"
    )


def _image_health(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if shutil.which("magick") is None:
        raise PipelineError("ImageMagick 7 (`magick`) is required")
    identify = subprocess.run(
        [
            "magick",
            "identify",
            "-format",
            "%w\t%h\t%m\t%[channels]",
            str(path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if identify.returncode:
        raise PipelineError(
            f"corrupt image: {identify.stderr.strip() or identify.stdout.strip()}"
        )
    parts = identify.stdout.split("\t")
    if len(parts) != 4:
        raise PipelineError("unexpected ImageMagick image-health output")
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as error:
        raise PipelineError("invalid image dimensions") from error
    source_format, source_mode = parts[2], parts[3]
    pixels_result = subprocess.run(
        [
            "magick",
            str(path),
            "-colorspace",
            "gray",
            "-resize",
            "9x8!",
            "-depth",
            "8",
            "gray:-",
        ],
        capture_output=True,
        check=False,
    )
    if pixels_result.returncode or len(pixels_result.stdout) != 72:
        raise PipelineError(
            "cannot derive image-health pixels: "
            + pixels_result.stderr.decode("utf-8", errors="replace").strip()
        )
    pixels = list(pixels_result.stdout)
    mean = sum(pixels) / len(pixels)
    stddev = (
        sum((value - mean) ** 2 for value in pixels) / len(pixels)
    ) ** 0.5
    minimum = int(config.get("minimum_dimension", 512))
    if min(width, height) < minimum:
        raise PipelineError(
            f"image dimensions {width}x{height} are below {minimum}"
        )
    if not source_mode or source_mode.casefold() in {"undefined", "unknown"}:
        raise PipelineError(f"unsupported image color mode: {source_mode}")
    blank_min = float(config.get("blank_stddev_min", 2.0))
    if stddev < blank_min:
        raise PipelineError(
            f"blank-image standard deviation {stddev:.4f} is below {blank_min}"
        )
    bits = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            bits = (bits << 1) | int(left > right)
    return {
        "width": width,
        "height": height,
        "mode": source_mode,
        "format": source_format,
        "grayscale_stddev": stddev,
        "perceptual_hash": f"{bits:016x}",
    }


def _hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def run_moondream_caption(
    path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise PipelineError(
            "Moondream dependencies are missing; install with "
            "`python3 -m pip install -e '.[captioning]'`"
        ) from error
    device = _local_device(torch, str(config.get("device", "auto")))
    key = _json_key(
        {
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "device": device,
        }
    )
    if _CAPTION_MODEL.get("key") != key:
        _CAPTION_MODEL.clear()
        dtype = torch.float32 if device == "cpu" else torch.float16
        try:
            model = AutoModelForCausalLM.from_pretrained(
                config["model_id"],
                revision=config["model_revision"],
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map={"": device},
            )
            model.eval()
        except Exception as error:
            raise PipelineError(
                f"cannot load Moondream captioner: {error}"
            ) from error
        _CAPTION_MODEL.update({"key": key, "model": model, "device": device})
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        try:
            torch.manual_seed(0)
            result = _CAPTION_MODEL["model"].caption(
                image,
                length=config["length"],
                settings={"temperature": config["temperature"]},
            )
        except Exception as error:
            raise PipelineError(f"Moondream captioning failed: {error}") from error
    raw = result.get("caption") if isinstance(result, dict) else result
    if not isinstance(raw, str) or not raw.strip():
        raise PipelineError("Moondream returned no caption")
    return {
        "raw_caption": raw.strip(),
        "runtime": {
            "torch": getattr(torch, "__version__", None),
            "transformers": _package_version("transformers"),
        },
        "device": device,
    }


def _clip_feature_tensor(value: Any, feature_name: str) -> Any:
    """Normalize Transformers 4.x tensors and 5.x pooled output wrappers."""
    pooled = getattr(value, "pooler_output", None)
    if pooled is not None:
        return pooled
    if isinstance(value, dict) and value.get("pooler_output") is not None:
        return value["pooler_output"]
    if hasattr(value, "norm"):
        return value
    raise PipelineError(
        f"CLIP {feature_name} features have unsupported type "
        f"{type(value).__name__}"
    )


def run_clip_qa(
    path: Path,
    caption: str | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as error:
        raise PipelineError(
            "alignment dependencies are missing; install with "
            "`python3 -m pip install -e '.[alignment]'`"
        ) from error
    device = _local_device(torch, str(config.get("device", "auto")))
    key = _json_key(
        {
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "device": device,
        }
    )
    if _CLIP_MODEL.get("key") != key:
        _CLIP_MODEL.clear()
        try:
            model = CLIPModel.from_pretrained(
                config["model_id"],
                revision=config["model_revision"],
            ).to(device)
            model.eval()
            processor = CLIPProcessor.from_pretrained(
                config["model_id"],
                revision=config["model_revision"],
            )
        except Exception as error:
            raise PipelineError(f"cannot load CLIP QA model: {error}") from error
        _CLIP_MODEL.update(
            {"key": key, "model": model, "processor": processor}
        )
    labels = [
        "a photograph of a real scene or object",
        "a painting, drawing, illustration, diagram, screenshot, or scanned page",
        "safe ordinary visual content",
        "graphic sexual, violent, hateful, or otherwise unsafe visual content",
    ]
    texts = ([caption] if caption is not None else []) + labels
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        processor = _CLIP_MODEL["processor"]
        model = _CLIP_MODEL["model"]
        inputs = processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            name: value.to(device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        try:
            with torch.inference_mode():
                image_features = _clip_feature_tensor(
                    model.get_image_features(
                        pixel_values=inputs["pixel_values"]
                    ),
                    "image",
                )
                text_features = _clip_feature_tensor(
                    model.get_text_features(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                    ),
                    "text",
                )
                image_features = image_features / image_features.norm(
                    dim=-1,
                    keepdim=True,
                )
                text_features = text_features / text_features.norm(
                    dim=-1,
                    keepdim=True,
                )
                similarities = (
                    image_features @ text_features.T
                )[0].detach().cpu()
        except Exception as error:
            raise PipelineError(f"CLIP QA inference failed: {error}") from error
    offset = 1 if caption is not None else 0
    photo_probability = torch.softmax(
        similarities[offset:offset + 2] * 100.0,
        dim=0,
    )[0].item()
    unsafe_probability = torch.softmax(
        similarities[offset + 2:offset + 4] * 100.0,
        dim=0,
    )[1].item()
    return {
        "alignment_score": (
            float(similarities[0].item()) if caption is not None else None
        ),
        "photo_probability": float(photo_probability),
        "unsafe_probability": float(unsafe_probability),
        "runtime": {
            "torch": getattr(torch, "__version__", None),
            "transformers": _package_version("transformers"),
        },
        "device": device,
    }


def _cached_json_result(
    cache_dir: Path,
    namespace: str,
    key: str,
    producer: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    path = cache_dir / namespace / f"{key}.json"
    if path.is_file():
        value = read_json(path)
        value["_cache_hit"] = True
        return value
    value = producer()
    if not isinstance(value, dict):
        raise PipelineError(f"{namespace} adapter returned a non-object")
    write_json(path, value)
    result = deepcopy(value)
    result["_cache_hit"] = False
    return result


def _concept_text(group: dict[str, Any]) -> str:
    concept = group.get("concept")
    if isinstance(concept, dict):
        value = concept.get("text")
        if isinstance(value, str) and value.strip():
            return value.strip().rstrip(".")
    prompt = group.get("prompt")
    text = prompt.get("text") if isinstance(prompt, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise PipelineError(
            f"{group.get('semantic_group_id', '<unknown>')}: "
            "a concept or prompt text is required"
        )
    return re.sub(
        r"^\s*a camera photograph of\s+",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).rstrip(".")


def _materialize_group(
    dataset: dict[str, Any],
    group: dict[str, Any],
) -> dict[str, Any]:
    prepared = deepcopy(group)
    concept_text = _concept_text(group)
    concept = group.get("concept")
    if isinstance(concept, dict):
        concept_record = deepcopy(concept)
    else:
        prompt = group.get("prompt", {})
        prompt_id = (
            prompt.get("prompt_id")
            if isinstance(prompt, dict)
            else None
        )
        concept_record = {
            "concept_id": prompt_id,
            "text": concept_text,
            "frozen": True,
        }
    prepared["concept"] = concept_record

    policy = dataset.get("prompt_policy")
    if policy is None:
        return prepared
    if not isinstance(policy, dict):
        raise PipelineError("dataset.prompt_policy must be an object")
    template_id = policy.get("template_id")
    template = policy.get("template")
    if not isinstance(template_id, str) or not template_id:
        raise PipelineError("dataset.prompt_policy.template_id is required")
    if not isinstance(template, str) or "{concept}" not in template:
        raise PipelineError(
            "dataset.prompt_policy.template must contain {concept}"
        )
    try:
        rendered = template.format(concept=concept_text).strip()
    except (KeyError, ValueError) as error:
        raise PipelineError(
            f"invalid dataset.prompt_policy.template: {error}"
        ) from error
    prompt = group.get("prompt")
    if not isinstance(prompt, dict):
        raise PipelineError(
            f"{group.get('semantic_group_id')}: prompt object is required"
        )
    prepared["prompt"] = {
        "prompt_id": prompt.get("prompt_id"),
        "text": rendered,
        "frozen": True,
    }
    prepared["prompt_policy"] = {
        "template_id": template_id,
        "template": template,
    }
    return prepared


def _review_policy(dataset: dict[str, Any]) -> dict[str, Any]:
    configured = dataset.get("generation_review", {})
    if not isinstance(configured, dict):
        raise PipelineError("dataset.generation_review must be an object")
    candidate_count = configured.get("candidates_per_slot", 1)
    if not isinstance(candidate_count, int) or not 1 <= candidate_count <= 16:
        raise PipelineError(
            "dataset.generation_review.candidates_per_slot must be 1..16"
        )
    require_explicit = configured.get("require_explicit_decision", False)
    if not isinstance(require_explicit, bool):
        raise PipelineError(
            "dataset.generation_review.require_explicit_decision must be boolean"
        )
    method = configured.get(
        "selection_method",
        "explicit-first-passing-v1" if require_explicit else "first-candidate-v1",
    )
    if not isinstance(method, str) or not method:
        raise PipelineError(
            "dataset.generation_review.selection_method is required"
        )
    reasons = configured.get(
        "allowed_rejection_reasons",
        sorted(DEFAULT_REJECTION_REASONS),
    )
    if (
        not isinstance(reasons, list)
        or not reasons
        or not all(isinstance(reason, str) and reason for reason in reasons)
    ):
        raise PipelineError(
            "dataset.generation_review.allowed_rejection_reasons "
            "must be a non-empty string array"
        )
    return {
        **configured,
        "candidates_per_slot": candidate_count,
        "require_explicit_decision": require_explicit,
        "selection_method": method,
        "allowed_rejection_reasons": reasons,
    }


def _json_key(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_map(values: Any) -> dict[str, Any]:
    if not isinstance(values, list):
        return {}
    return {
        str(item["name"]): item.get("value")
        for item in values
        if isinstance(item, dict) and "name" in item
    }


def _extended_map(values: Any) -> dict[str, Any]:
    if not isinstance(values, dict):
        return {}
    result: dict[str, Any] = {}
    for key, item in values.items():
        result[key] = item.get("value") if isinstance(item, dict) else item
    return result


def _request_json(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 180,
) -> tuple[dict[str, Any], dict[str, str]]:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
        **(headers or {}),
    }
    payload = None
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = Request(
        url,
        data=payload,
        headers=request_headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
            response_headers = {
                key.casefold(): value for key, value in response.headers.items()
            }
    except HTTPError as error:
        detail = error.read(2048).decode("utf-8", errors="replace")
        raise PipelineError(f"HTTP {error.code} from {url}: {detail}") from error
    except (OSError, URLError) as error:
        raise PipelineError(f"request failed for {url}: {error}") from error
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise PipelineError(f"non-JSON response from {url}") from error
    if not isinstance(result, dict):
        raise PipelineError(f"unexpected response from {url}")
    return result, response_headers


def _download(url: str, target: Path, *, expected_sha256: str | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.download")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    total = 0
    digest = hashlib.sha256()
    try:
        with urlopen(request, timeout=180) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise PipelineError(
                        f"download exceeds {MAX_DOWNLOAD_BYTES} bytes: {url}"
                    )
                digest.update(chunk)
                output.write(chunk)
    except PipelineError:
        temporary.unlink(missing_ok=True)
        raise
    except HTTPError as error:
        temporary.unlink(missing_ok=True)
        detail = error.read(2048).decode("utf-8", errors="replace")
        raise PipelineError(
            f"HTTP {error.code} while downloading {url}: {detail}"
        ) from error
    except (OSError, URLError) as error:
        temporary.unlink(missing_ok=True)
        raise PipelineError(f"download failed for {url}: {error}") from error
    observed = digest.hexdigest()
    if expected_sha256 and observed.casefold() != expected_sha256.casefold():
        temporary.unlink(missing_ok=True)
        raise PipelineError(
            f"download checksum mismatch for {url}: "
            f"expected {expected_sha256}, observed {observed}"
        )
    os.replace(temporary, target)


def _require_jpeg(path: Path) -> None:
    with path.open("rb") as handle:
        signature = handle.read(len(JPEG_SIGNATURE))
    if signature != JPEG_SIGNATURE:
        path.unlink(missing_ok=True)
        raise PipelineError(f"camera source is not a JPEG: {path}")


def _commons_candidate(page: dict[str, Any]) -> dict[str, Any] | None:
    image_info = page.get("imageinfo")
    if not isinstance(image_info, list) or not image_info:
        return None
    info = image_info[0]
    if not isinstance(info, dict) or info.get("mime") != "image/jpeg":
        return None
    if min(info.get("width", 0), info.get("height", 0)) < 512:
        return None
    metadata = _metadata_map(info.get("metadata"))
    extended = _extended_map(info.get("extmetadata"))
    make = metadata.get("Make")
    model = metadata.get("Model")
    captured_at = (
        metadata.get("DateTimeOriginal")
        or metadata.get("DateTimeDigitized")
        or metadata.get("DateTime")
    )
    if not all(isinstance(value, str) and value.strip() for value in (
        make,
        model,
        captured_at,
    )):
        return None
    software = str(metadata.get("Software", ""))
    if any(editor in software.casefold() for editor in REJECTED_EDITORS):
        return None
    license_name = str(extended.get("LicenseShortName", ""))
    license_url = str(extended.get("LicenseUrl", ""))
    allowed_license = (
        license_name.casefold().startswith("cc ")
        or license_name.casefold() in {"cc0", "public domain"}
    )
    if not allowed_license or not license_url:
        return None
    return {
        "page": page,
        "info": info,
        "metadata": metadata,
        "extended": extended,
        "capture": {
            "camera_make": make.strip(),
            "camera_model": model.strip(),
            "captured_at": captured_at.strip(),
            "edit_screen_status": "pass",
            "software": software or None,
        },
        "license": {"name": license_name, "url": license_url},
    }


def _camera_search_queries(group: dict[str, Any]) -> list[str]:
    configured = group.get("camera_source", {}).get("query")
    raw = (
        configured
        if isinstance(configured, str) and configured.strip()
        else _concept_text(group)
    )
    tokens = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", raw.casefold())
    semantic = [token for token in tokens if token not in QUERY_SCAFFOLDING]
    filtered = [token for token in tokens if token not in SEARCH_STOPWORDS]
    if not filtered:
        filtered = tokens
    queries = [" ".join(semantic)] if semantic else []
    compact = " ".join(filtered)
    if compact and compact not in queries:
        queries.append(compact)
    for length in (len(filtered), 6, 5, 4, 3, 2):
        if length > len(filtered) or length < 1:
            continue
        query = " ".join(filtered[:length])
        if query and query not in queries:
            queries.append(query)
    return queries


def fetch_wikimedia_camera(
    group: dict[str, Any],
    config: dict[str, Any],
    target: Path,
) -> dict[str, Any]:
    endpoint = str(
        config.get(
            "api_url",
            "https://commons.wikimedia.org/w/api.php",
        )
    )
    queries = _camera_search_queries(group)
    candidates: list[tuple[str, dict[str, Any]]] = []
    seen_pages: set[str] = set()
    candidate_limit = config.get("candidate_limit", 8)
    if not isinstance(candidate_limit, int) or not 1 <= candidate_limit <= 40:
        raise PipelineError("camera candidate_limit must be 1..40")
    for query in queries:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrnamespace": "6",
            "gsrlimit": str(config.get("search_limit", 40)),
            "gsrsearch": f"{query} filetype:bitmap",
            "prop": "info|imageinfo",
            "inprop": "url",
            "iiprop": "url|size|mime|sha1|metadata|extmetadata",
        }
        response, _ = _request_json(f"{endpoint}?{urlencode(params)}")
        pages = response.get("query", {}).get("pages", [])
        if not isinstance(pages, list):
            pages = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            accepted = _commons_candidate(page)
            page_id = str(page.get("pageid"))
            if accepted is None or page_id in seen_pages:
                continue
            candidates.append((query, accepted))
            seen_pages.add(page_id)
            if len(candidates) >= candidate_limit:
                break
        if candidates:
            break
    if not candidates:
        raise PipelineError(
            f"{group['semantic_group_id']}: Wikimedia search returned no "
            "license-compatible JPEG with camera EXIF and no detected editor; "
            f"queries={queries}"
        )
    selected_index = config.get("search_candidate_index", 0)
    if (
        not isinstance(selected_index, int)
        or selected_index < 0
        or selected_index >= len(candidates)
    ):
        raise PipelineError(
            f"{group['semantic_group_id']}: camera search_candidate_index "
            f"{selected_index!r} is outside 0..{len(candidates) - 1}"
        )
    selected_query, candidate = candidates[selected_index]
    page = candidate["page"]
    info = candidate["info"]
    _download(info["url"], target)
    _require_jpeg(target)
    page_id = str(page.get("pageid"))
    landing_page = (
        page.get("canonicalurl")
        or page.get("fullurl")
        or f"https://commons.wikimedia.org/?curid={page_id}"
    )
    source = {
        "collection_id": "wikimedia-commons",
        "source_record_id": page_id,
        "landing_page_url": landing_page,
        "license": candidate["license"],
        "creator": candidate["extended"].get("Artist"),
        "credit": candidate["extended"].get("Credit"),
    }
    provenance = {
        "kind": "wikimedia-commons-acquisition",
        "acquired_at": utc_now(),
        "query": selected_query,
        "attempted_queries": queries,
        "candidate_count": len(candidates),
        "selected_candidate_index": selected_index,
        "candidates": [
            {
                "candidate_index": index,
                "page_id": str(item["page"].get("pageid")),
                "query": query,
                "title": item["page"].get("title"),
                "width": item["info"].get("width"),
                "height": item["info"].get("height"),
            }
            for index, (query, item) in enumerate(candidates)
        ],
        "page_id": page_id,
        "title": page.get("title"),
        "original_url": info["url"],
        "commons_sha1": info.get("sha1"),
        "width": info.get("width"),
        "height": info.get("height"),
        "mime": info.get("mime"),
        "artist": candidate["extended"].get("Artist"),
        "credit": candidate["extended"].get("Credit"),
        "camera_metadata": candidate["metadata"],
    }
    return {
        "source": source,
        "capture": candidate["capture"],
        "generation": None,
        "scope": {"in_scope": True, "ambiguity_flags": []},
        "audit": {
            "selection_method": "wikimedia-search-camera-exif-v2",
            "automated_edit_screen": True,
            "candidate_index": selected_index,
            "candidate_count": len(candidates),
        },
        "provenance": provenance,
    }


def fetch_direct_camera(
    group: dict[str, Any],
    config: dict[str, Any],
    target: Path,
) -> dict[str, Any]:
    source_config = group.get("camera_source")
    if not isinstance(source_config, dict):
        raise PipelineError(
            f"{group['semantic_group_id']}: camera_source object is required"
        )
    for field in (
        "url",
        "sha256",
        "source_record_id",
        "landing_page_url",
        "license",
        "capture",
    ):
        if source_config.get(field) in (None, ""):
            raise PipelineError(
                f"{group['semantic_group_id']}: camera_source.{field} is required"
            )
    _download(
        source_config["url"],
        target,
        expected_sha256=source_config["sha256"],
    )
    _require_jpeg(target)
    capture = deepcopy(source_config["capture"])
    capture["edit_screen_status"] = "pass"
    return {
        "source": {
            "collection_id": source_config.get("collection_id", "direct-url"),
            "source_record_id": source_config["source_record_id"],
            "landing_page_url": source_config["landing_page_url"],
            "license": source_config["license"],
            "creator": source_config.get("creator"),
            "credit": source_config.get("credit"),
        },
        "capture": capture,
        "generation": None,
        "scope": {"in_scope": True, "ambiguity_flags": []},
        "audit": {"selection_method": "pinned-direct-url"},
        "provenance": {
            "kind": "direct-url-acquisition",
            "acquired_at": utc_now(),
            "url": source_config["url"],
            "declared_sha256": source_config["sha256"],
        },
    }


_LOCAL_PIPELINE: dict[str, Any] = {}
_LOCAL_MODEL_REVISIONS: dict[tuple[str, str | None], str] = {}


def _local_device(torch: Any, configured: str) -> str:
    if configured != "auto":
        return configured
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _release_local_pipeline(torch: Any) -> None:
    _LOCAL_PIPELINE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_local_diffusers(
    group: dict[str, Any],
    config: dict[str, Any],
    seed: int | None,
    target: Path,
) -> dict[str, Any]:
    try:
        import torch
        from diffusers import DiffusionPipeline
        from huggingface_hub import HfApi
    except ImportError as error:
        raise PipelineError(
            "local generation dependencies are missing; install with "
            "`python3 -m pip install -e '.[generation]'`"
        ) from error
    if seed is None:
        raise PipelineError("local generation requires a deterministic seed")

    model_id = str(config["model_id"])
    requested_revision = config.get("model_revision")
    revision_key = (
        model_id,
        str(requested_revision) if requested_revision else None,
    )
    model_revision = _LOCAL_MODEL_REVISIONS.get(revision_key)
    if model_revision is None:
        print(f"[local model] resolve {model_id}", flush=True)
        try:
            resolved_revision = HfApi().model_info(
                model_id,
                revision=requested_revision,
            ).sha
        except Exception as error:
            raise PipelineError(
                f"cannot resolve Hugging Face revision for {model_id}: {error}"
            ) from error
        if not isinstance(resolved_revision, str) or not resolved_revision:
            raise PipelineError(
                f"Hugging Face returned no commit revision for {model_id}"
            )
        model_revision = resolved_revision
        _LOCAL_MODEL_REVISIONS[revision_key] = model_revision

    device = _local_device(torch, str(config.get("device", "auto")))
    if device == "cpu":
        dtype = torch.float32
    elif device.startswith("cuda") and getattr(
        torch.cuda,
        "is_bf16_supported",
        lambda: False,
    )():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    pipeline_key = json.dumps(
        {
            "model_id": model_id,
            "model_revision": model_revision,
            "device": device,
            "dtype": str(dtype),
        },
        sort_keys=True,
    )
    if _LOCAL_PIPELINE.get("key") != pipeline_key:
        _release_local_pipeline(torch)
        print(
            f"[local model] load {model_id}@{model_revision[:12]} "
            f"on {device} ({dtype})",
            flush=True,
        )
        try:
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                revision=model_revision,
                torch_dtype=dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
            if device.startswith("cuda") and bool(
                config.get("cpu_offload", True)
            ):
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(device)
            pipeline.set_progress_bar_config(disable=True)
        except Exception as error:
            _release_local_pipeline(torch)
            raise PipelineError(
                f"cannot load local model {model_id}: {error}"
            ) from error
        _LOCAL_PIPELINE.update({"key": pipeline_key, "pipeline": pipeline})
    pipeline = _LOCAL_PIPELINE["pipeline"]

    settings = deepcopy(config.get("settings", {}))
    kwargs = {
        key: value
        for key, value in settings.items()
        if key in {
            "negative_prompt",
            "height",
            "width",
            "num_inference_steps",
            "guidance_scale",
            "max_sequence_length",
        }
    }
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scheduler = getattr(pipeline, "scheduler", None)
    scheduler_config = getattr(scheduler, "config", {})
    try:
        scheduler_config = json.loads(
            json.dumps(dict(scheduler_config), default=str)
        )
    except (TypeError, ValueError):
        scheduler_config = {"value": str(scheduler_config)}
    try:
        result = pipeline(
            group["prompt"]["text"],
            generator=generator,
            **kwargs,
        )
        image = result.images[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(target, format="PNG")
    except Exception as error:
        target.unlink(missing_ok=True)
        raise PipelineError(
            f"local generation failed for {model_id}: {error}"
        ) from error
    return {
        "capture": None,
        "generation": {
            "family_id": config["family_id"],
            "model_id": model_id,
            "model_revision": model_revision,
            "provider": "local-diffusers",
            "settings": settings,
            "input_image_used": False,
            "seed_status": "recorded",
            "seed": seed,
            "rendered_prompt": group["prompt"]["text"],
            "prompt_template_id": group.get("prompt_policy", {}).get(
                "template_id"
            ),
        },
        "scope": {
            "in_scope": False,
            "ambiguity_flags": ["pending-review"],
        },
        "audit": {"review_status": "pending"},
        "provenance": {
            "kind": "local-diffusers-generation",
            "generated_at": utc_now(),
            "execution_device": device,
            "torch_dtype": str(dtype),
            "pipeline_class": type(pipeline).__name__,
            "scheduler_class": (
                type(scheduler).__name__ if scheduler is not None else None
            ),
            "scheduler_config": scheduler_config,
            "model": model_id,
            "model_revision": model_revision,
            "prompt": group["prompt"],
            "concept": group.get("concept"),
            "prompt_policy": group.get("prompt_policy"),
            "settings": {**settings, "seed": seed},
            "runtime": {
                "diffusers": _package_version("diffusers"),
                "huggingface_hub": _package_version("huggingface-hub"),
                "torch": getattr(torch, "__version__", None),
                "transformers": _package_version("transformers"),
            },
        },
    }


def _seed(seed_base: int, group_id: str, slot_id: str) -> int:
    digest = hashlib.sha256(
        f"{seed_base}:{group_id}:{slot_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _candidate_seed(
    seed_base: int,
    group_id: str,
    slot_id: str,
    candidate_index: int,
) -> int:
    if candidate_index == 0:
        return _seed(seed_base, group_id, slot_id)
    return _seed(
        seed_base,
        group_id,
        f"{slot_id}-candidate-{candidate_index}",
    )


def _review_paths(cache_dir: Path) -> tuple[Path, Path]:
    review_dir = cache_dir / "review"
    return review_dir / "candidates.json", review_dir / "decisions.json"


def _load_review_decisions(cache_dir: Path) -> dict[str, Any]:
    _, decisions_path = _review_paths(cache_dir)
    if not decisions_path.is_file():
        return {}
    document = read_json(decisions_path)
    decisions = document.get("decisions")
    if not isinstance(decisions, dict):
        raise PipelineError(
            f"{decisions_path}: decisions must be an object"
        )
    return decisions


def _review_decision(
    sample_id: str,
    candidates: list[dict[str, Any]],
    policy: dict[str, Any],
    decisions: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    decision = decisions.get(sample_id)
    if (
        not policy["require_explicit_decision"]
        and (
            decision is None
            or (
                isinstance(decision, dict)
                and decision.get("status") == "pending"
            )
        )
    ):
        decision = {
            "status": "accepted",
            "candidate_index": 0,
            "rejected_candidates": {},
            "reviewer": None,
            "reviewed_at": None,
        }
    if not isinstance(decision, dict):
        return None
    if decision.get("status") != "accepted":
        return None
    selected_index = decision.get("candidate_index")
    if (
        not isinstance(selected_index, int)
        or selected_index < 0
        or selected_index >= len(candidates)
    ):
        raise PipelineError(
            f"{sample_id}: review candidate_index must be within "
            f"0..{len(candidates) - 1}"
        )
    rejected = decision.get("rejected_candidates", {})
    if not isinstance(rejected, dict):
        raise PipelineError(
            f"{sample_id}: rejected_candidates must be an object"
        )
    if policy["require_explicit_decision"]:
        for field in ("reviewer", "reviewed_at"):
            if not isinstance(decision.get(field), str) or not decision[field]:
                raise PipelineError(
                    f"{sample_id}: explicit review requires {field}"
                )
        selected_sha256 = decision.get("candidate_sha256")
        if selected_sha256 != candidates[selected_index]["sha256"]:
            raise PipelineError(
                f"{sample_id}: candidate_sha256 does not match candidate "
                f"{selected_index}; review the current candidate bytes"
            )
    for raw_index in rejected:
        try:
            rejected_index = int(raw_index)
        except (TypeError, ValueError) as error:
            raise PipelineError(
                f"{sample_id}: rejected candidate keys must be integers"
            ) from error
        if str(rejected_index) != raw_index or not 0 <= rejected_index < selected_index:
            raise PipelineError(
                f"{sample_id}: rejected candidate {raw_index!r} does not "
                "precede the accepted candidate"
            )
    allowed = set(policy["allowed_rejection_reasons"])
    for index in range(selected_index):
        reasons = rejected.get(str(index))
        if (
            not isinstance(reasons, list)
            or not reasons
            or not all(isinstance(reason, str) for reason in reasons)
        ):
            raise PipelineError(
                f"{sample_id}: candidate {index} must have rejection reasons "
                "before a later candidate can be accepted"
            )
        unknown = sorted(set(reasons) - allowed)
        if unknown:
            raise PipelineError(
                f"{sample_id}: candidate {index} has unknown rejection "
                f"reasons: {', '.join(unknown)}"
            )
    return candidates[selected_index], deepcopy(decision)


def _write_review_documents(
    cache_dir: Path,
    policy: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
    decisions: dict[str, Any],
) -> tuple[Path, Path]:
    candidates_path, decisions_path = _review_paths(cache_dir)
    write_json(
        candidates_path,
        {
            "schema_version": "1.0.0",
            "selection_method": policy["selection_method"],
            "allowed_rejection_reasons": policy["allowed_rejection_reasons"],
            "candidates": candidates,
        },
    )
    scaffold = deepcopy(decisions)
    for sample_id in candidates:
        scaffold.setdefault(
            sample_id,
            {
                "status": (
                    "pending"
                    if policy["require_explicit_decision"]
                    else "accepted"
                ),
                "candidate_index": (
                    None if policy["require_explicit_decision"] else 0
                ),
                "candidate_sha256": None,
                "rejected_candidates": {},
                "reviewer": None,
                "reviewed_at": None,
            },
        )
    write_json(
        decisions_path,
        {
            "schema_version": "1.0.0",
            "selection_method": policy["selection_method"],
            "decisions": scaffold,
        },
    )
    sections: list[str] = []
    for sample_id, sample_candidates in sorted(candidates.items()):
        cards = []
        for candidate in sample_candidates:
            relative_image = "../" + candidate["path"]
            cards.append(
                "<figure>"
                f'<img src="{escape(relative_image)}" '
                'loading="lazy" width="256" height="256">'
                f"<figcaption>candidate {candidate['candidate_index']} · "
                f"seed {candidate['seed']}</figcaption>"
                "</figure>"
            )
        prompt = (
            sample_candidates[0].get("prompt", "")
            if sample_candidates
            else ""
        )
        sections.append(
            f"<section><h2>{escape(sample_id)}</h2>"
            f"<p>{escape(str(prompt))}</p>"
            f'<div class="candidates">{"".join(cards)}</div></section>'
        )
    review_html = candidates_path.with_name("index.html")
    temporary = review_html.with_name(f".{review_html.name}.tmp")
    temporary.write_text(
        (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            "<title>MAI generation review</title><style>"
            "body{font:14px system-ui;margin:2rem;background:#111;color:#eee}"
            "section{margin-bottom:2rem}.candidates{display:flex;gap:1rem;"
            "flex-wrap:wrap}figure{margin:0;background:#222;padding:.6rem}"
            "img{object-fit:contain;background:#000;display:block}"
            "figcaption{margin-top:.5rem}</style></head><body>"
            "<h1>MAI generation candidates</h1>"
            "<p>Record decisions in <code>decisions.json</code>.</p>"
            + "".join(sections)
            + "</body></html>"
        ),
        encoding="utf-8",
    )
    os.replace(temporary, review_html)
    return candidates_path, decisions_path


def _cache_entry_valid(
    cache_dir: Path | None,
    cache_key: str,
    suffix: str,
) -> bool:
    if cache_dir is None:
        return False
    asset = cache_dir / "assets" / f"{cache_key}{suffix}"
    receipt_path = cache_dir / "receipts" / f"{cache_key}.json"
    if not asset.is_file() or not receipt_path.is_file():
        return False
    try:
        receipt = read_json(receipt_path)
        return (
            receipt.get("asset_sha256") == sha256(asset)
            and isinstance(receipt.get("sample_fields"), dict)
        )
    except (OSError, PipelineError):
        return False


def _real_photo_plan(
    spec: dict[str, Any],
    groups: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset = spec["dataset"]
    expected_slots = dataset["expected_slots"]
    synthetic_slots = [
        slot
        for slot in expected_slots
        if slot["origin_class"] == "synthetic"
    ]
    selection = dataset["generation_selection"]
    candidates = selection["candidates_per_slot"]
    highest_source_index = max(
        (int(group["source_index"]) for group in groups),
        default=-1,
    )
    source = dataset["real_source"]
    oversample_factor = int(source.get("oversample_factor", 20))
    return {
        "pipeline": "hf-real-photo-caption-v2",
        "group_count": len(groups),
        "sample_count": len(groups) * len(expected_slots),
        "cache_hits": 0,
        "camera_downloads": 0,
        "source_rows_requested": (
            (highest_source_index + 1) * oversample_factor
        ),
        "caption_jobs": len(groups),
        "generation_jobs": len(groups) * len(synthetic_slots) * candidates,
        "generation_candidate_count": (
            len(groups) * len(synthetic_slots) * candidates
        ),
        "generation_candidates_per_slot": candidates,
        "qa_jobs": (
            2 * len(groups)
            + len(groups) * len(synthetic_slots) * candidates
        ),
        "review_decisions_pending": 0,
        "configured_credentials": [],
        "required_credentials": [],
        "missing_credentials": [],
        "accepted_group_count": len(groups),
        "quarantined_group_count": 0,
        "quarantine_files": [],
    }


def _source_rejection(
    cache_dir: Path,
    source_id: str,
    reason: str,
    details: dict[str, Any],
) -> None:
    key = _json_key({"source_id": source_id, "reason": reason, **details})
    write_json(
        cache_dir / "source-rejections" / f"{key}.json",
        {
            "schema_version": "1.0.0",
            "status": "rejected-before-split",
            "source_record_id": source_id,
            "reason": reason,
            "details": details,
        },
    )


def _qa_failures(
    qa: dict[str, Any],
    config: dict[str, Any],
    *,
    require_alignment: bool,
) -> list[str]:
    failures: list[str] = []
    if require_alignment:
        score = qa.get("alignment_score")
        threshold = float(config["alignment_threshold"])
        if not isinstance(score, (int, float)) or float(score) < threshold:
            failures.append("semantic_alignment")
    photo_min = float(config.get("photo_probability_min", 0.0))
    photo_score = qa.get("photo_probability")
    if (
        photo_min > 0
        and (
            not isinstance(photo_score, (int, float))
            or float(photo_score) < photo_min
        )
    ):
        failures.append("non_photographic_style")
    unsafe_max = float(config.get("unsafe_probability_max", 1.0))
    unsafe_score = qa.get("unsafe_probability")
    if (
        unsafe_max < 1
        and (
            not isinstance(unsafe_score, (int, float))
            or float(unsafe_score) > unsafe_max
        )
    ):
        failures.append("unsafe_content")
    return failures


def _select_hf_sources(
    dataset: dict[str, Any],
    groups: list[dict[str, Any]],
    cache_dir: Path,
    source_loader: RealSourceLoader,
    qa_runner: QARunner,
) -> dict[int, dict[str, Any]]:
    config = dataset["real_source"]
    qa_config = dataset["automated_qa"]
    highest_index = max(int(group["source_index"]) for group in groups)
    needed = highest_index + 1
    limit = needed * int(config.get("oversample_factor", 20))
    rows = source_loader(config, limit)
    if not isinstance(rows, list):
        rows = list(rows)
    selected: list[dict[str, Any]] = []
    exact_digests: dict[str, str] = {}
    perceptual_hashes: list[tuple[str, str]] = []
    image_column = str(config["image_column"])
    id_column = str(config["id_column"])
    license_column = str(config["license_column"])
    source_url_column = str(config["source_url_column"])
    allowed_mime = set(config.get("allowed_mime_types", []))
    metadata_columns = config.get("metadata_filter_columns", [])
    reject_patterns = [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in config.get("metadata_reject_patterns", [])
    ]
    for rank, row in enumerate(rows):
        if len(selected) >= needed:
            break
        if not isinstance(row, dict):
            continue
        source_id = row.get(id_column)
        if source_id in (None, ""):
            _source_rejection(
                cache_dir,
                f"row-{rank}",
                "missing_stable_id",
                {"rank": rank},
            )
            continue
        source_id = str(source_id)
        canonical_license = _canonical_license(
            row.get(license_column),
            config["license_policy"],
        )
        if canonical_license is None:
            _source_rejection(
                cache_dir,
                source_id,
                "license_not_allowed",
                {"license": row.get(license_column)},
            )
            continue
        source_url = row.get(source_url_column)
        if not isinstance(source_url, str) or not source_url:
            _source_rejection(
                cache_dir,
                source_id,
                "missing_source_url",
                {},
            )
            continue
        mime_column = config.get("mime_type_column")
        mime_type = row.get(mime_column) if isinstance(mime_column, str) else None
        if allowed_mime and mime_type not in allowed_mime:
            _source_rejection(
                cache_dir,
                source_id,
                "invalid_mime_type",
                {"mime_type": mime_type},
            )
            continue
        width_column = config.get("width_column")
        height_column = config.get("height_column")
        declared_width = (
            row.get(width_column) if isinstance(width_column, str) else None
        )
        declared_height = (
            row.get(height_column) if isinstance(height_column, str) else None
        )
        minimum = int(qa_config["minimum_dimension"])
        if (
            isinstance(declared_width, int)
            and isinstance(declared_height, int)
            and min(declared_width, declared_height) < minimum
        ):
            _source_rejection(
                cache_dir,
                source_id,
                "declared_dimensions_too_small",
                {
                    "width": declared_width,
                    "height": declared_height,
                },
            )
            continue
        metadata_text = " ".join(
            str(row.get(column, ""))
            for column in metadata_columns
            if isinstance(column, str)
        )
        matched = next(
            (
                pattern.pattern
                for pattern in reject_patterns
                if pattern.search(metadata_text)
            ),
            None,
        )
        if matched is not None:
            _source_rejection(
                cache_dir,
                source_id,
                "metadata_content_filter",
                {"matched_pattern": matched},
            )
            continue
        image_value = row.get(image_column)
        cache_key = _json_key(
            {
                "kind": "hf-real-photo",
                "dataset_id": config["dataset_id"],
                "revision": config["revision"],
                "split": config["split"],
                "source_record_id": source_id,
                "source_url": source_url,
            }
        )

        def produce_source(
            target: Path,
            value: Any = image_value,
        ) -> dict[str, Any]:
            _materialize_hf_image(value, target)
            health = _image_health(target, qa_config)
            name = canonical_license[0]
            source_name_column = config.get("source_name_column")
            source_name = (
                row.get(source_name_column)
                if isinstance(source_name_column, str)
                else None
            )
            return {
                "source": {
                    "collection_id": config["dataset_id"],
                    "source_record_id": source_id,
                    "landing_page_url": source_url,
                    "license": {
                        "name": name,
                        "url": canonical_license[1],
                    },
                    "source_name": source_name,
                },
                "capture": None,
                "generation": None,
                "scope": {"in_scope": True, "ambiguity_flags": []},
                "audit": {
                    "selection_method": "seeded-hf-stream-first-valid-v1",
                    "source_rank": rank,
                },
                "provenance": {
                    "kind": "huggingface-real-photo-acquisition",
                    "real_source": {
                        "dataset_id": config["dataset_id"],
                        "dataset_revision": config["revision"],
                        "source_split": config["split"],
                        "source_row_id": source_id,
                        "source_url": source_url,
                        "source_name": source_name,
                        "declared_license": row.get(license_column),
                    },
                    "image_health": health,
                    "runtime": {
                        "datasets": _package_version("datasets"),
                        "pillow": _package_version("Pillow"),
                    },
                },
            }

        try:
            fields = _cached_sample(
                cache_dir,
                cache_key,
                ".img",
                produce_source,
            )
            path = Path(fields["input_path"])
            health = fields["provenance"]["image_health"]
            digest = sha256(path)
        except (OSError, PipelineError) as error:
            _source_rejection(
                cache_dir,
                source_id,
                "corrupt_or_invalid_image",
                {"error": str(error)},
            )
            continue
        if digest in exact_digests:
            _source_rejection(
                cache_dir,
                source_id,
                "exact_duplicate",
                {"duplicates_source_record_id": exact_digests[digest]},
            )
            continue
        perceptual_hash = health["perceptual_hash"]
        max_distance = int(qa_config["near_duplicate_hamming_distance"])
        near_duplicate = next(
            (
                previous_id
                for previous_hash, previous_id in perceptual_hashes
                if _hamming_distance(perceptual_hash, previous_hash)
                <= max_distance
            ),
            None,
        )
        if near_duplicate is not None:
            _source_rejection(
                cache_dir,
                source_id,
                "perceptual_near_duplicate",
                {"duplicates_source_record_id": near_duplicate},
            )
            continue
        qa_key = _json_key(
            {
                "kind": "source-photo-screen",
                "sha256": digest,
                "qa": qa_config,
            }
        )
        try:
            qa = _cached_json_result(
                cache_dir,
                "qa",
                qa_key,
                lambda: qa_runner(path, None, qa_config),
            )
        except Exception as error:
            raise PipelineError(
                f"automated source QA failed for {source_id}: {error}"
            ) from error
        failures = _qa_failures(
            qa,
            qa_config,
            require_alignment=False,
        )
        if failures:
            _source_rejection(
                cache_dir,
                source_id,
                "automated_content_filter",
                {"failures": failures, "scores": qa},
            )
            continue
        fields["provenance"]["original_sha256"] = digest
        fields["provenance"]["original_bytes"] = path.stat().st_size
        fields["provenance"]["source_qa"] = {
            **qa,
            "model_id": qa_config["model_id"],
            "model_revision": qa_config["model_revision"],
        }
        exact_digests[digest] = source_id
        perceptual_hashes.append((perceptual_hash, source_id))
        selected.append(fields)
    if len(selected) < needed:
        raise PipelineError(
            f"HF source produced {len(selected)} valid unique photos for "
            f"{needed} deterministic source positions; increase "
            "real_source.oversample_factor or revise the filters"
        )
    return {index: fields for index, fields in enumerate(selected)}


def _human_audit_required(
    group_id: str,
    seed: int,
    rate: float,
) -> bool:
    value = int.from_bytes(
        hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()[:8],
        "big",
    )
    return value / float(2**64) < rate


def _quarantine_group(
    cache_dir: Path,
    group: dict[str, Any],
    stage: str,
    reason: str,
    details: dict[str, Any],
) -> Path:
    path = cache_dir / "quarantine" / (
        f"{group['semantic_group_id']}.json"
    )
    write_json(
        path,
        {
            "schema_version": "1.0.0",
            "status": "quarantined",
            "semantic_group_id": group["semantic_group_id"],
            "source_index": group["source_index"],
            "split": group["split"],
            "stage": stage,
            "reason": reason,
            "manual_override_status": "none",
            "details": details,
        },
    )
    return path


def preparation_plan(
    spec: dict[str, Any],
    groups: list[dict[str, Any]],
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    dataset, _, _, _ = validate_dataset_design(spec)
    if isinstance(dataset.get("real_source"), dict):
        return _real_photo_plan(spec, groups)
    prepared_groups = [
        _materialize_group(dataset, group) for group in groups
    ]
    review_policy = _review_policy(dataset)
    candidate_count = review_policy["candidates_per_slot"]
    review_decisions = (
        _load_review_decisions(cache_dir)
        if cache_dir is not None
        else {}
    )
    expected_slots = dataset.get("expected_slots", [])
    synthetic_slots = [
        slot for slot in expected_slots
        if isinstance(slot, dict) and slot.get("origin_class") == "synthetic"
    ]
    generators = dataset.get("generators")
    if not isinstance(generators, dict):
        raise PipelineError("dataset.generators object is required")
    configured_credentials: set[str] = set()
    for slot in synthetic_slots:
        family = slot.get("generator_family")
        config = generators.get(family)
        if not isinstance(config, dict):
            raise PipelineError(f"no generator configured for family {family}")
        if config.get("family_id") != family:
            raise PipelineError(f"generator configuration differs for {family}")
        for field in ("adapter", "model_id", "settings", "output_terms_url"):
            if config.get(field) in (None, ""):
                raise PipelineError(f"{family}.{field} is required")
        if (
            review_policy["require_explicit_decision"]
            and not config.get("model_revision")
        ):
            raise PipelineError(
                f"{family}.model_revision is required when explicit "
                "generation review is enabled"
            )
        credential = config.get("credential_env")
        if isinstance(credential, str) and credential:
            configured_credentials.add(credential)
    acquisition = dataset.get("camera_acquisition")
    if not isinstance(acquisition, dict) or not acquisition.get("adapter"):
        raise PipelineError("dataset.camera_acquisition.adapter is required")
    seed_base = int(dataset.get("seed_base", 1729))
    cache_hits = 0
    camera_downloads = 0
    generation_jobs = 0
    required_credentials: set[str] = set()
    review_decisions_pending = 0
    for group in prepared_groups:
        for slot in expected_slots:
            if slot["origin_class"] == "camera":
                source_config = {
                    **acquisition,
                    **(
                        group.get("camera_source", {})
                        if isinstance(group.get("camera_source"), dict)
                        else {}
                    ),
                }
                cache_key = _json_key({
                    "kind": "camera",
                    "group": group,
                    "config": source_config,
                })
                cached = _cache_entry_valid(cache_dir, cache_key, ".jpg")
                if not cached:
                    camera_downloads += 1
            else:
                family = slot["generator_family"]
                generator_config = generators[family]
                for candidate_index in range(candidate_count):
                    seed = _candidate_seed(
                        seed_base,
                        group["semantic_group_id"],
                        slot["slot_id"],
                        candidate_index,
                    )
                    cache_key = _json_key({
                        "kind": "synthetic",
                        "group": group,
                        "slot": slot,
                        "config": generator_config,
                        "candidate_index": candidate_index,
                        "seed": seed,
                    })
                    cached = _cache_entry_valid(
                        cache_dir,
                        cache_key,
                        ".png",
                    )
                    if not cached:
                        generation_jobs += 1
                        credential = generator_config.get("credential_env")
                        if isinstance(credential, str) and credential:
                            required_credentials.add(credential)
                    else:
                        cache_hits += 1
                sample_id = (
                    f"{group['semantic_group_id']}-{slot['slot_id']}"
                )
                decision = review_decisions.get(sample_id)
                if (
                    review_policy["require_explicit_decision"]
                    and (
                        not isinstance(decision, dict)
                        or decision.get("status") != "accepted"
                    )
                ):
                    review_decisions_pending += 1
                continue
            if cached:
                cache_hits += 1
    return {
        "group_count": len(groups),
        "sample_count": len(groups) * len(expected_slots),
        "cache_hits": cache_hits,
        "camera_downloads": camera_downloads,
        "generation_jobs": generation_jobs,
        "generation_candidate_count": (
            len(prepared_groups) * len(synthetic_slots) * candidate_count
        ),
        "generation_candidates_per_slot": candidate_count,
        "review_decisions_pending": review_decisions_pending,
        "configured_credentials": sorted(configured_credentials),
        "required_credentials": sorted(required_credentials),
        "missing_credentials": sorted(
            name for name in required_credentials if not os.environ.get(name)
        ),
    }


def _cached_sample(
    cache_dir: Path,
    cache_key: str,
    suffix: str,
    producer: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    asset = cache_dir / "assets" / f"{cache_key}{suffix}"
    receipt_path = cache_dir / "receipts" / f"{cache_key}.json"
    if asset.is_file() and receipt_path.is_file():
        receipt = read_json(receipt_path)
        if (
            receipt.get("asset_sha256") == sha256(asset)
            and isinstance(receipt.get("sample_fields"), dict)
        ):
            fields = deepcopy(receipt["sample_fields"])
            fields["input_path"] = str(asset)
            fields.setdefault("audit", {})["cache_hit"] = True
            return fields
    fields = producer(asset)
    if not asset.is_file():
        raise PipelineError("preparation adapter produced no image")
    fields.setdefault("audit", {})["cache_hit"] = False
    write_json(
        receipt_path,
        {
            "cache_schema": "1.0.0",
            "asset_sha256": sha256(asset),
            "asset_bytes": asset.stat().st_size,
            "sample_fields": fields,
        },
    )
    result = deepcopy(fields)
    result["input_path"] = str(asset)
    return result


def _prepare_real_photo_groups(
    spec: dict[str, Any],
    groups: list[dict[str, Any]],
    cache_dir: Path,
    *,
    source_loaders: dict[str, RealSourceLoader] | None,
    caption_runners: dict[str, CaptionRunner] | None,
    qa_runners: dict[str, QARunner] | None,
    generator_runners: dict[str, GeneratorRunner] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = spec["dataset"]
    plan = _real_photo_plan(spec, groups)
    source_config = dataset["real_source"]
    caption_config = dataset["captioning"]
    qa_config = dataset["automated_qa"]
    selection = dataset["generation_selection"]
    expected_slots = dataset["expected_slots"]
    generators = dataset["generators"]
    source_loaders = source_loaders or {
        "huggingface-dataset": load_huggingface_rows,
    }
    caption_runners = caption_runners or {
        "moondream-local": run_moondream_caption,
    }
    qa_runners = qa_runners or {"clip-local": run_clip_qa}
    generator_runners = generator_runners or {
        "local-diffusers": run_local_diffusers,
    }
    source_loader = source_loaders.get(source_config["adapter"])
    if source_loader is None:
        raise PipelineError(
            f"unknown real source adapter: {source_config['adapter']}"
        )
    caption_runner = caption_runners.get(caption_config["adapter"])
    if caption_runner is None:
        raise PipelineError(
            f"unknown caption adapter: {caption_config['adapter']}"
        )
    qa_runner = qa_runners.get(qa_config["adapter"])
    if qa_runner is None:
        raise PipelineError(
            f"unknown automated QA adapter: {qa_config['adapter']}"
        )
    source_by_index = _select_hf_sources(
        dataset,
        groups,
        cache_dir,
        source_loader,
        qa_runner,
    )
    real_slot = next(
        slot for slot in expected_slots
        if slot["origin_class"] == "real_photo"
    )
    synthetic_slots = [
        slot
        for slot in expected_slots
        if slot["origin_class"] == "synthetic"
    ]
    candidate_count = int(selection["candidates_per_slot"])
    seed_base = int(dataset.get("seed_base", 1729))
    audit_rate = float(selection["human_audit_rate"])
    samples: list[dict[str, Any]] = []
    quarantine_files: list[str] = []
    cache_hits = 0
    generated_candidate = 0
    for group_number, configured_group in enumerate(groups, 1):
        group = {
            **configured_group,
            "content_category": configured_group.get(
                "content_category",
                "unstratified",
            ),
        }
        group_id = group["semantic_group_id"]
        stale_quarantine = cache_dir / "quarantine" / f"{group_id}.json"
        source_fields = deepcopy(source_by_index[int(group["source_index"])])
        source_path = Path(source_fields["input_path"])
        source_digest = sha256(source_path)
        caption_key = _json_key(
            {
                "kind": "caption",
                "source_sha256": source_digest,
                "captioning": caption_config,
            }
        )
        try:
            def produce_caption() -> dict[str, Any]:
                value = caption_runner(source_path, caption_config)
                if isinstance(value, str):
                    return {"raw_caption": value}
                if not isinstance(value, dict):
                    raise PipelineError(
                        "caption adapter returned neither text nor an object"
                    )
                return value

            caption_result = _cached_json_result(
                cache_dir,
                "captions",
                caption_key,
                produce_caption,
            )
            cache_hits += int(caption_result.pop("_cache_hit", False))
            raw_caption = caption_result.get("raw_caption")
            if not isinstance(raw_caption, str):
                raw_caption = caption_result.get("caption")
            normalized_caption = normalize_caption(
                raw_caption,
                caption_config["normalization_policy"],
            )
            prompt = _render_caption_prompt(
                group_id,
                normalized_caption,
                dataset["prompt_policy"],
            )
        except Exception as error:
            quarantine = _quarantine_group(
                cache_dir,
                group,
                "captioning",
                "caption_exception",
                {
                    "error": str(error),
                    "source_record_id": source_fields["source"][
                        "source_record_id"
                    ],
                    "source_sha256": source_digest,
                },
            )
            quarantine_files.append(str(quarantine))
            continue
        group.update(
            {
                "prompt": prompt,
                "concept": {
                    "concept_id": prompt["prompt_id"],
                    "text": normalized_caption.rstrip("."),
                    "frozen": True,
                },
                "prompt_policy": deepcopy(dataset["prompt_policy"]),
            }
        )
        real_qa_key = _json_key(
            {
                "kind": "caption-alignment",
                "source_sha256": source_digest,
                "caption": normalized_caption,
                "qa": qa_config,
            }
        )
        try:
            real_qa = _cached_json_result(
                cache_dir,
                "qa",
                real_qa_key,
                lambda: qa_runner(
                    source_path,
                    normalized_caption,
                    qa_config,
                ),
            )
            cache_hits += int(real_qa.pop("_cache_hit", False))
        except Exception as error:
            quarantine = _quarantine_group(
                cache_dir,
                group,
                "real-image-qa",
                "qa_exception",
                {"error": str(error), "caption": normalized_caption},
            )
            quarantine_files.append(str(quarantine))
            continue
        real_failures = _qa_failures(
            real_qa,
            qa_config,
            require_alignment=True,
        )
        if real_failures:
            quarantine = _quarantine_group(
                cache_dir,
                group,
                "real-image-qa",
                "real_image_failed_qa",
                {
                    "failures": real_failures,
                    "scores": real_qa,
                    "caption": normalized_caption,
                },
            )
            quarantine_files.append(str(quarantine))
            continue
        audit_required = _human_audit_required(
            group_id,
            seed_base,
            audit_rate,
        )
        caption_lineage = {
            "raw_caption": raw_caption.strip(),
            "normalized_caption": normalized_caption,
            "caption_policy_version": caption_config[
                "normalization_policy"
            ],
            "model_id": caption_config["model_id"],
            "model_revision": caption_config["model_revision"],
            "settings": {
                "length": caption_config["length"],
                "temperature": caption_config["temperature"],
            },
            "runtime": caption_result.get("runtime", {}),
        }
        source_lineage = {
            "source_photo_group_id": group_id,
            "hf_dataset_id": source_config["dataset_id"],
            "hf_dataset_revision": source_config["revision"],
            "source_split": source_config["split"],
            "source_row_id": source_fields["source"]["source_record_id"],
            "source_url": source_fields["source"]["landing_page_url"],
            "original_sha256": source_digest,
            "original_bytes": source_path.stat().st_size,
        }
        source_fields["provenance"].update(
            {
                "captioning": caption_lineage,
                "lineage": source_lineage,
                "quarantine_status": "accepted",
                "manual_override_status": "none",
                "automated_qa": {
                    "model_id": qa_config["model_id"],
                    "model_revision": qa_config["model_revision"],
                    "thresholds": {
                        "alignment": qa_config["alignment_threshold"],
                        "photo_probability_min": qa_config.get(
                            "photo_probability_min"
                        ),
                        "unsafe_probability_max": qa_config.get(
                            "unsafe_probability_max"
                        ),
                    },
                    "scores": real_qa,
                },
            }
        )
        source_fields.setdefault("audit", {}).update(
            {
                "selection_method": "seeded-hf-stream-first-valid-v1",
                "human_audit_required": audit_required,
                "human_audit_status": "pending" if audit_required else "not-sampled",
                "manual_override_status": "none",
            }
        )
        real_sample = {
            "semantic_group_id": group_id,
            "content_category": group["content_category"],
            "split": group["split"],
            "prompt": prompt,
            **source_fields,
            "sample_id": f"{group_id}-{real_slot['slot_id']}",
            "slot_id": real_slot["slot_id"],
            "origin_class": "real_photo",
        }
        group_samples = [real_sample]
        group_failure: tuple[str, dict[str, Any]] | None = None
        for slot in synthetic_slots:
            slot_id = slot["slot_id"]
            family = slot["generator_family"]
            generator_config = generators[family]
            runner = generator_runners.get(generator_config["adapter"])
            if runner is None:
                raise PipelineError(
                    f"unknown generator adapter: {generator_config['adapter']}"
                )
            sample_id = f"{group_id}-{slot_id}"
            candidate_records: list[dict[str, Any]] = []
            passing_fields: dict[int, dict[str, Any]] = {}
            for candidate_index in range(candidate_count):
                seed = _candidate_seed(
                    seed_base,
                    group_id,
                    slot_id,
                    candidate_index,
                )
                cache_key = _json_key(
                    {
                        "kind": "synthetic-v2",
                        "group": group,
                        "source_lineage": source_lineage,
                        "slot": slot,
                        "config": generator_config,
                        "candidate_index": candidate_index,
                        "seed": seed,
                    }
                )
                generated_candidate += 1
                try:
                    fields = _cached_sample(
                        cache_dir,
                        cache_key,
                        ".png",
                        lambda target, candidate_seed=seed: runner(
                            group,
                            generator_config,
                            candidate_seed,
                            target,
                        ),
                    )
                    cache_hits += int(
                        fields.get("audit", {}).get("cache_hit") is True
                    )
                    candidate_path = Path(fields["input_path"])
                    health = _image_health(candidate_path, qa_config)
                    candidate_digest = sha256(candidate_path)
                    qa_key = _json_key(
                        {
                            "kind": "generated-caption-alignment",
                            "sha256": candidate_digest,
                            "caption": normalized_caption,
                            "qa": qa_config,
                        }
                    )
                    candidate_qa = _cached_json_result(
                        cache_dir,
                        "qa",
                        qa_key,
                        lambda: qa_runner(
                            candidate_path,
                            normalized_caption,
                            qa_config,
                        ),
                    )
                    cache_hits += int(
                        candidate_qa.pop("_cache_hit", False)
                    )
                    failures = _qa_failures(
                        candidate_qa,
                        qa_config,
                        require_alignment=True,
                    )
                    candidate_record = {
                        "candidate_index": candidate_index,
                        "seed": seed,
                        "sha256": candidate_digest,
                        "bytes": candidate_path.stat().st_size,
                        "image_health": health,
                        "qa_scores": candidate_qa,
                        "failures": failures,
                        "passed": not failures,
                    }
                    if not failures:
                        passing_fields[candidate_index] = fields
                except Exception as error:
                    candidate_record = {
                        "candidate_index": candidate_index,
                        "seed": seed,
                        "passed": False,
                        "failures": ["exception"],
                        "error": str(error),
                    }
                candidate_records.append(candidate_record)
            selected_candidate = next(
                (
                    candidate
                    for candidate in candidate_records
                    if candidate.get("passed") is True
                ),
                None,
            )
            if selected_candidate is None:
                group_failure = (
                    slot_id,
                    {
                        "reason": "all_generated_candidates_failed",
                        "candidates": candidate_records,
                    },
                )
                break
            selected_index = int(selected_candidate["candidate_index"])
            fields = passing_fields[selected_index]
            fields["scope"] = {"in_scope": True, "ambiguity_flags": []}
            fields.setdefault("generation", {}).update(
                {
                    "candidate_index": selected_index,
                    "candidate_count": candidate_count,
                    "rendered_prompt": prompt["text"],
                    "prompt_template_id": dataset["prompt_policy"][
                        "template_id"
                    ],
                }
            )
            fields.setdefault("audit", {}).update(
                {
                    "selection_method": selection["method"],
                    "review_status": "not-required",
                    "candidate_index": selected_index,
                    "candidate_count": candidate_count,
                    "candidate_qa": candidate_records,
                    "human_audit_required": audit_required,
                    "human_audit_status": (
                        "pending" if audit_required else "not-sampled"
                    ),
                    "manual_override_status": "none",
                }
            )
            fields.setdefault("provenance", {}).update(
                {
                    "lineage": source_lineage,
                    "captioning": caption_lineage,
                    "automated_qa": {
                        "model_id": qa_config["model_id"],
                        "model_revision": qa_config["model_revision"],
                        "thresholds": {
                            "alignment": qa_config[
                                "alignment_threshold"
                            ],
                            "photo_probability_min": qa_config.get(
                                "photo_probability_min"
                            ),
                            "unsafe_probability_max": qa_config.get(
                                "unsafe_probability_max"
                            ),
                        },
                        "selected_scores": selected_candidate["qa_scores"],
                    },
                    "candidate_index": selected_index,
                    "candidate_count": candidate_count,
                    "selection_method": selection["method"],
                    "candidates": candidate_records,
                    "quarantine_status": "accepted",
                    "manual_override_status": "none",
                }
            )
            fields["source"] = {
                "collection_id": f"mai-{family}-generation",
                "source_record_id": sample_id,
                "landing_page_url": (
                    f"https://huggingface.co/{generator_config['model_id']}"
                ),
                "license": {
                    "name": "model-output-license",
                    "url": generator_config["output_terms_url"],
                },
            }
            group_samples.append(
                {
                    "semantic_group_id": group_id,
                    "content_category": group["content_category"],
                    "split": group["split"],
                    "prompt": prompt,
                    **fields,
                    "sample_id": sample_id,
                    "slot_id": slot_id,
                    "origin_class": "synthetic",
                }
            )
        if group_failure is not None:
            failed_slot, failure_details = group_failure
            quarantine = _quarantine_group(
                cache_dir,
                group,
                "generation-qa",
                failure_details["reason"],
                {
                    "failed_slot": failed_slot,
                    "caption": normalized_caption,
                    "source_lineage": source_lineage,
                    **failure_details,
                },
            )
            quarantine_files.append(str(quarantine))
            continue
        stale_quarantine.unlink(missing_ok=True)
        samples.extend(group_samples)
        print(
            f"[prepare {group_number}/{len(groups)}] {group_id} "
            f"({len(group_samples)} accepted samples)",
            flush=True,
        )
    plan.update(
        {
            "cache_hits": cache_hits,
            "accepted_group_count": len(samples) // len(expected_slots),
            "quarantined_group_count": len(quarantine_files),
            "quarantine_files": quarantine_files,
        }
    )
    return samples, plan


def prepare_groups(
    spec: dict[str, Any],
    groups: list[dict[str, Any]],
    cache_dir: Path,
    *,
    camera_fetchers: dict[str, CameraFetcher] | None = None,
    generator_runners: dict[str, GeneratorRunner] | None = None,
    source_loaders: dict[str, RealSourceLoader] | None = None,
    caption_runners: dict[str, CaptionRunner] | None = None,
    qa_runners: dict[str, QARunner] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(spec.get("dataset", {}).get("real_source"), dict):
        return _prepare_real_photo_groups(
            spec,
            groups,
            cache_dir,
            source_loaders=source_loaders,
            caption_runners=caption_runners,
            qa_runners=qa_runners,
            generator_runners=generator_runners,
        )
    plan = preparation_plan(spec, groups, cache_dir)
    if plan["missing_credentials"]:
        raise PipelineError(
            "missing required credentials: "
            + ", ".join(plan["missing_credentials"])
        )
    dataset = spec["dataset"]
    groups = [_materialize_group(dataset, group) for group in groups]
    expected_slots = dataset["expected_slots"]
    generators = dataset["generators"]
    acquisition = dataset["camera_acquisition"]
    camera_fetchers = camera_fetchers or {
        "wikimedia-commons": fetch_wikimedia_camera,
        "direct-url": fetch_direct_camera,
    }
    generator_runners = generator_runners or {
        "local-diffusers": run_local_diffusers,
    }
    seed_base = int(dataset.get("seed_base", 1729))
    review_policy = _review_policy(dataset)
    candidate_count = review_policy["candidates_per_slot"]
    review_decisions = _load_review_decisions(cache_dir)
    review_candidates: dict[str, list[dict[str, Any]]] = {}
    pending_review: list[str] = []
    samples: list[dict[str, Any]] = []
    operation_count = plan["sample_count"]
    operation = 0
    generated_candidate = 0

    def append_sample(
        group: dict[str, Any],
        slot: dict[str, Any],
        fields: dict[str, Any],
    ) -> None:
        nonlocal operation
        operation += 1
        group_id = group["semantic_group_id"]
        slot_id = slot["slot_id"]
        sample_id = f"{group_id}-{slot_id}"
        sample = {
            "semantic_group_id": group_id,
            "content_category": group["content_category"],
            "split": group["split"],
            "prompt": group["prompt"],
            **fields,
            "sample_id": sample_id,
            "slot_id": slot_id,
            "origin_class": slot["origin_class"],
        }
        samples.append(sample)
        cache_status = (
            "cache" if sample.get("audit", {}).get("cache_hit") else "network"
        )
        print(
            f"[prepare {operation}/{operation_count}] "
            f"{sample_id} ({cache_status})",
            flush=True,
        )

    camera_slot = next(
        slot for slot in expected_slots if slot["origin_class"] == "camera"
    )
    camera_records: dict[tuple[str, str], str] = {}
    camera_digests: dict[str, str] = {}
    for group in groups:
        group_id = group["semantic_group_id"]
        source_config = {
            **acquisition,
            **(
                group.get("camera_source", {})
                if isinstance(group.get("camera_source"), dict)
                else {}
            ),
        }
        adapter = source_config["adapter"]
        fetcher = camera_fetchers.get(adapter)
        if fetcher is None:
            raise PipelineError(f"unknown camera adapter: {adapter}")
        cache_key = _json_key({
            "kind": "camera",
            "group": group,
            "config": source_config,
        })
        fields = _cached_sample(
            cache_dir,
            cache_key,
            ".jpg",
            lambda target: fetcher(group, source_config, target),
        )
        source = fields.get("source", {})
        record_key = (
            str(source.get("collection_id")),
            str(source.get("source_record_id")),
        )
        previous_group = camera_records.get(record_key)
        if previous_group is not None:
            raise PipelineError(
                f"{group_id}: camera source duplicates {previous_group}; "
                "set a different camera_source.query or pin a direct source"
            )
        digest = sha256(Path(fields["input_path"]))
        previous_group = camera_digests.get(digest)
        if previous_group is not None:
            raise PipelineError(
                f"{group_id}: camera bytes duplicate {previous_group}; "
                "set a different camera_source.query or pin a direct source"
            )
        camera_records[record_key] = group_id
        camera_digests[digest] = group_id
        append_sample(group, camera_slot, fields)

    synthetic_slots = [
        slot for slot in expected_slots if slot["origin_class"] == "synthetic"
    ]
    families = list(dict.fromkeys(
        slot["generator_family"] for slot in synthetic_slots
    ))
    for family in families:
        family_slots = [
            slot
            for slot in synthetic_slots
            if slot["generator_family"] == family
        ]
        for group in groups:
            group_id = group["semantic_group_id"]
            for slot in family_slots:
                slot_id = slot["slot_id"]
                generator_config = generators[family]
                adapter = generator_config["adapter"]
                runner = generator_runners.get(adapter)
                if runner is None:
                    raise PipelineError(f"unknown generator adapter: {adapter}")
                sample_id = f"{group_id}-{slot_id}"
                candidates: list[dict[str, Any]] = []
                candidate_fields: list[dict[str, Any]] = []
                for candidate_index in range(candidate_count):
                    seed = _candidate_seed(
                        seed_base,
                        group_id,
                        slot_id,
                        candidate_index,
                    )
                    cache_key = _json_key({
                        "kind": "synthetic",
                        "group": group,
                        "slot": slot,
                        "config": generator_config,
                        "candidate_index": candidate_index,
                        "seed": seed,
                    })
                    generated_candidate += 1
                    if not _cache_entry_valid(cache_dir, cache_key, ".png"):
                        print(
                            f"[generate {generated_candidate}/"
                            f"{plan['generation_candidate_count']}] "
                            f"{sample_id} candidate {candidate_index} locally "
                            f"with {generator_config['model_id']}",
                            flush=True,
                        )
                    fields = _cached_sample(
                        cache_dir,
                        cache_key,
                        ".png",
                        lambda target, candidate_seed=seed: runner(
                            group,
                            generator_config,
                            candidate_seed,
                            target,
                        ),
                    )
                    candidate_fields.append(fields)
                    candidate_path = Path(fields["input_path"])
                    candidates.append(
                        {
                            "candidate_index": candidate_index,
                            "seed": seed,
                            "path": candidate_path.relative_to(
                                cache_dir
                            ).as_posix(),
                            "sha256": sha256(candidate_path),
                            "bytes": candidate_path.stat().st_size,
                            "prompt": group["prompt"]["text"],
                            "prompt_template_id": group.get(
                                "prompt_policy", {}
                            ).get("template_id"),
                            "model_id": generator_config["model_id"],
                            "model_revision": fields.get(
                                "generation", {}
                            ).get("model_revision"),
                        }
                    )
                review_candidates[sample_id] = candidates
                selected = _review_decision(
                    sample_id,
                    candidates,
                    review_policy,
                    review_decisions,
                )
                if selected is None:
                    pending_review.append(sample_id)
                    continue
                selected_candidate, decision = selected
                selected_index = selected_candidate["candidate_index"]
                fields = candidate_fields[selected_index]
                fields["scope"] = {
                    "in_scope": True,
                    "ambiguity_flags": [],
                }
                fields.setdefault("generation", {}).update(
                    {
                        "candidate_index": selected_index,
                        "candidate_count": candidate_count,
                        "rendered_prompt": group["prompt"]["text"],
                        "prompt_template_id": group.get(
                            "prompt_policy", {}
                        ).get("template_id"),
                    }
                )
                fields.setdefault("audit", {}).update(
                    {
                        "selection_method": review_policy["selection_method"],
                        "review_status": "accepted",
                        "reviewer": decision.get("reviewer"),
                        "reviewed_at": decision.get("reviewed_at"),
                        "candidate_index": selected_index,
                        "candidate_count": candidate_count,
                        "rejected_candidates": decision.get(
                            "rejected_candidates", {}
                        ),
                    }
                )
                fields.setdefault("provenance", {}).update(
                    {
                        "candidate_index": selected_index,
                        "candidate_count": candidate_count,
                        "review": decision,
                    }
                )
                fields["source"] = {
                    "collection_id": f"mai-{family}-generation",
                    "source_record_id": sample_id,
                    "landing_page_url": (
                        f"https://huggingface.co/{generator_config['model_id']}"
                    ),
                    "license": {
                        "name": "model-output-license",
                        "url": generator_config["output_terms_url"],
                    },
                }
                append_sample(group, slot, fields)
    if review_candidates:
        _, decisions_path = _write_review_documents(
            cache_dir,
            review_policy,
            review_candidates,
            review_decisions,
        )
    else:
        decisions_path = _review_paths(cache_dir)[1]
    if pending_review:
        raise PipelineError(
            f"{len(pending_review)} synthetic slots require review; inspect "
            f"{_review_paths(cache_dir)[0]} and complete {decisions_path}, "
            "then run Build again (generated candidates will be reused)"
        )
    return samples, plan
