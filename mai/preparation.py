"""On-demand camera acquisition and synthetic-image generation."""

from __future__ import annotations

from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import re
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


USER_AGENT = "mai-research/0.2 (+https://github.com/kenneth/mai)"
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

CameraFetcher = Callable[
    [dict[str, Any], dict[str, Any], Path],
    dict[str, Any],
]
GeneratorRunner = Callable[
    [dict[str, Any], dict[str, Any], int | None, Path],
    dict[str, Any],
]


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
    prompt = group["prompt"]["text"]
    configured = group.get("camera_source", {}).get("query")
    raw = (
        configured
        if isinstance(configured, str) and configured.strip()
        else re.sub(
            r"^\s*a camera photograph of\s+",
            "",
            prompt,
            flags=re.IGNORECASE,
        ).rstrip(".")
    )
    tokens = re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", raw.casefold())
    filtered = [token for token in tokens if token not in SEARCH_STOPWORDS]
    if not filtered:
        filtered = tokens
    queries: list[str] = []
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
    candidate = None
    selected_query = None
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
        candidate = next(
            (
                accepted
                for page in pages
                if isinstance(page, dict)
                for accepted in [_commons_candidate(page)]
                if accepted is not None
            ),
            None,
        )
        if candidate is not None:
            selected_query = query
            break
    if candidate is None:
        raise PipelineError(
            f"{group['semantic_group_id']}: Wikimedia search returned no "
            "license-compatible JPEG with camera EXIF and no detected editor; "
            f"queries={queries}"
        )
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
    }
    provenance = {
        "kind": "wikimedia-commons-acquisition",
        "acquired_at": utc_now(),
        "query": selected_query,
        "attempted_queries": queries,
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
            "selection_method": "wikimedia-search-first-camera-exif-v1",
            "automated_edit_screen": True,
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
        },
        "scope": {"in_scope": True, "ambiguity_flags": []},
        "audit": {},
        "provenance": {
            "kind": "local-diffusers-generation",
            "generated_at": utc_now(),
            "execution_device": device,
            "torch_dtype": str(dtype),
            "pipeline_class": type(pipeline).__name__,
            "model": model_id,
            "model_revision": model_revision,
            "prompt": group["prompt"],
            "settings": {**settings, "seed": seed},
        },
    }


def _seed(seed_base: int, group_id: str, slot_id: str) -> int:
    digest = hashlib.sha256(
        f"{seed_base}:{group_id}:{slot_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


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


def preparation_plan(
    spec: dict[str, Any],
    groups: list[dict[str, Any]],
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    dataset, _, _, _ = validate_dataset_design(spec)
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
    for group in groups:
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
                adapter = generator_config["adapter"]
                seed = _seed(
                    seed_base,
                    group["semantic_group_id"],
                    slot["slot_id"],
                )
                cache_key = _json_key({
                    "kind": "synthetic",
                    "group": group,
                    "slot": slot,
                    "config": generator_config,
                    "seed": seed,
                })
                cached = _cache_entry_valid(cache_dir, cache_key, ".png")
                if not cached:
                    generation_jobs += 1
                    credential = generator_config.get("credential_env")
                    if isinstance(credential, str) and credential:
                        required_credentials.add(credential)
            if cached:
                cache_hits += 1
    return {
        "group_count": len(groups),
        "sample_count": len(groups) * len(expected_slots),
        "cache_hits": cache_hits,
        "camera_downloads": camera_downloads,
        "generation_jobs": generation_jobs,
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


def prepare_groups(
    spec: dict[str, Any],
    groups: list[dict[str, Any]],
    cache_dir: Path,
    *,
    camera_fetchers: dict[str, CameraFetcher] | None = None,
    generator_runners: dict[str, GeneratorRunner] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = preparation_plan(spec, groups, cache_dir)
    if plan["missing_credentials"]:
        raise PipelineError(
            "missing required credentials: "
            + ", ".join(plan["missing_credentials"])
        )
    dataset = spec["dataset"]
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
    samples: list[dict[str, Any]] = []
    operation_count = plan["sample_count"]
    operation = 0

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
                seed = _seed(seed_base, group_id, slot_id)
                cache_key = _json_key({
                    "kind": "synthetic",
                    "group": group,
                    "slot": slot,
                    "config": generator_config,
                    "seed": seed,
                })
                if not _cache_entry_valid(cache_dir, cache_key, ".png"):
                    print(
                        f"[generate {operation + 1}/{operation_count}] "
                        f"{group_id}-{slot_id} locally with "
                        f"{generator_config['model_id']}",
                        flush=True,
                    )
                fields = _cached_sample(
                    cache_dir,
                    cache_key,
                    ".png",
                    lambda target: runner(
                        group,
                        generator_config,
                        seed,
                        target,
                    ),
                )
                sample_id = f"{group_id}-{slot_id}"
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
    return samples, plan
