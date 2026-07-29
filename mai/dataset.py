"""Build and validate a Hugging Face-native MAI dataset package."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
from typing import Any, Iterable

from . import __version__


BUILD_SPEC_SCHEMA_VERSION = "2.0.0"
PACKAGE_SCHEMA_VERSION = "3.0.0"
LEGACY_PACKAGE_SCHEMA_VERSION = "2.0.0"
SUPPORTED_PACKAGE_SCHEMA_VERSIONS = {
    LEGACY_PACKAGE_SCHEMA_VERSION,
    PACKAGE_SCHEMA_VERSION,
}
PARQUET_SHARD_TARGET_BYTES = 256 * 1024 * 1024
PARQUET_ROW_GROUP_SIZE = 100
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
NORMALIZATION = {
    "transformation_id": "normalize-imagemagick-v1",
    "tool": "ImageMagick",
    "algorithm": (
        "apply orientation; convert to sRGB; resize to cover 512x512; "
        "center crop; strip metadata; encode non-interlaced 8-bit RGB PNG"
    ),
    "width": 512,
    "height": 512,
    "media_type": "image/png",
    "color_space": "sRGB",
    "channels": 3,
    "bit_depth": 8,
}


class PipelineError(RuntimeError):
    """A user-actionable dataset pipeline failure."""


class Validation:
    def __init__(self, schema_version: str = PACKAGE_SCHEMA_VERSION) -> None:
        self.schema_version = schema_version
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.checks = 0

    def require(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.errors.append(message)

    def warn_unless(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.warnings.append(message)

    def report(self, **summary: Any) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": "pass" if not self.errors else "fail",
            "checks": self.checks,
            "errors": self.errors,
            "warnings": self.warnings,
            **summary,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PipelineError(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise PipelineError(f"cannot read JSONL {path}: {error}") from error
    with handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise PipelineError(f"{path}:{number}: {error}") from error
            if not isinstance(value, dict):
                raise PipelineError(f"{path}:{number}: expected a JSON object")
            records.append(value)
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
            )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise PipelineError(f"path escapes package root: {relative}") from error
    return candidate


def require_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise PipelineError(
            f"{field} must match {ID_PATTERN.pattern!r}; received {value!r}"
        )
    return value


def load_spec(spec_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = read_json(spec_path)
    if spec.get("schema_version") != BUILD_SPEC_SCHEMA_VERSION:
        raise PipelineError(
            "build spec schema_version must be "
            f"{BUILD_SPEC_SCHEMA_VERSION!r}"
        )
    samples = spec.get("samples")
    samples_file = spec.get("samples_file")
    groups = spec.get("groups")
    configured_sources = sum(
        value is not None for value in (samples, samples_file, groups)
    )
    if configured_sources != 1:
        raise PipelineError(
            "use exactly one of groups, samples, or samples_file"
        )
    if groups is not None:
        if not isinstance(groups, list) or not all(
            isinstance(group, dict) for group in groups
        ):
            raise PipelineError("build spec groups must be an array of objects")
        samples = []
        inherited_fields = (
            "semantic_group_id",
            "content_category",
            "split",
            "prompt",
        )
        for group in groups:
            group_id = group.get("semantic_group_id", "<missing>")
            group_samples = group.get("samples", [])
            if not isinstance(group_samples, list) or not all(
                isinstance(sample, dict) for sample in group_samples
            ):
                raise PipelineError(f"{group_id}: samples must be an array")
            for sample in group_samples:
                expanded = dict(sample)
                for field in inherited_fields:
                    if field not in group:
                        raise PipelineError(f"{group_id}: {field} is required")
                    if field in expanded and expanded[field] != group[field]:
                        raise PipelineError(
                            f"{group_id}: sample overrides group-level {field}"
                        )
                    expanded[field] = group[field]
                samples.append(expanded)
    if samples_file is not None:
        if not isinstance(samples_file, str):
            raise PipelineError("samples_file must be a relative path")
        samples = read_jsonl((spec_path.parent / samples_file).resolve())
    if not isinstance(samples, list) or not all(
        isinstance(sample, dict) for sample in samples
    ):
        raise PipelineError("build spec must contain a samples array or samples_file")
    return spec, samples


def sample_input_path(spec_path: Path, sample: dict[str, Any]) -> Path:
    value = sample.get("input_path")
    if not isinstance(value, str) or not value:
        raise PipelineError(f"{sample.get('sample_id')}: input_path is required")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (spec_path.parent / path).resolve()


def validate_dataset_design(
    spec: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    set[str],
    int,
]:
    dataset = spec.get("dataset")
    if not isinstance(dataset, dict):
        raise PipelineError("build spec dataset object is required")
    require_id(dataset.get("dataset_id"), "dataset.dataset_id")
    for field in ("title", "description", "license"):
        if not isinstance(dataset.get(field), str) or not dataset[field].strip():
            raise PipelineError(f"dataset.{field} is required")
    target_group_count = dataset.get("target_group_count")
    if not isinstance(target_group_count, int) or target_group_count < 1:
        raise PipelineError("dataset.target_group_count must be a positive integer")
    expected_slots = dataset.get("expected_slots")
    if not isinstance(expected_slots, list) or not expected_slots:
        raise PipelineError("dataset.expected_slots must be a non-empty array")
    slot_by_id: dict[str, dict[str, Any]] = {}
    for slot in expected_slots:
        if not isinstance(slot, dict):
            raise PipelineError("every expected slot must be an object")
        slot_id = require_id(slot.get("slot_id"), "expected slot_id")
        if slot_id in slot_by_id:
            raise PipelineError(f"duplicate expected slot: {slot_id}")
        origin = slot.get("origin_class")
        if origin not in {"camera", "synthetic"}:
            raise PipelineError(f"{slot_id}: invalid expected origin_class")
        if origin == "synthetic":
            require_id(slot.get("generator_family"), f"{slot_id}.generator_family")
        slot_by_id[slot_id] = slot
    camera_slots = [
        slot for slot in expected_slots if slot["origin_class"] == "camera"
    ]
    if len(camera_slots) != 1:
        raise PipelineError("expected_slots must contain exactly one camera slot")
    families = {
        slot["generator_family"]
        for slot in expected_slots
        if slot["origin_class"] == "synthetic"
    }
    if len(families) < 2:
        raise PipelineError("expected_slots must contain at least two generator families")
    return dataset, slot_by_id, families, target_group_count


def validate_spec(
    spec_path: Path,
    spec: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset, slot_by_id, families, target_group_count = (
        validate_dataset_design(spec)
    )
    expected_slots = dataset["expected_slots"]
    if not samples:
        raise PipelineError("build spec contains no samples")

    ids: set[str] = set()
    matrix: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    group_prompts: dict[str, set[tuple[str, str]]] = defaultdict(set)
    group_categories: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        sample_id = require_id(sample.get("sample_id"), "sample_id")
        if sample_id in ids:
            raise PipelineError(f"duplicate sample_id: {sample_id}")
        ids.add(sample_id)
        group_id = require_id(sample.get("semantic_group_id"), "semantic_group_id")
        slot_id = require_id(sample.get("slot_id"), f"{sample_id}.slot_id")
        if slot_id not in slot_by_id:
            raise PipelineError(f"{sample_id}: unknown slot_id {slot_id}")
        if slot_id in matrix[group_id]:
            raise PipelineError(f"{group_id}: duplicate slot {slot_id}")
        matrix[group_id][slot_id] = sample
        expected = slot_by_id[slot_id]
        origin = sample.get("origin_class")
        if origin != expected["origin_class"]:
            raise PipelineError(f"{sample_id}: origin_class differs from its slot")
        prompt = sample.get("prompt")
        if (
            not isinstance(prompt, dict)
            or not isinstance(prompt.get("prompt_id"), str)
            or not isinstance(prompt.get("text"), str)
            or not prompt["text"]
            or prompt.get("frozen") is not True
        ):
            raise PipelineError(f"{sample_id}: a frozen prompt is required")
        require_id(prompt["prompt_id"], f"{sample_id}.prompt.prompt_id")
        group_prompts[group_id].add((prompt["prompt_id"], prompt["text"]))
        category = sample.get("content_category")
        if not isinstance(category, str) or not category:
            raise PipelineError(f"{sample_id}: content_category is required")
        group_categories[group_id].add(category)
        split = sample.get("split", "train")
        require_id(split, f"{sample_id}.split")
        group_splits[group_id].add(split)
        source = sample.get("source")
        if not isinstance(source, dict):
            raise PipelineError(f"{sample_id}: source provenance is required")
        for field in ("collection_id", "source_record_id", "landing_page_url"):
            if not isinstance(source.get(field), str) or not source[field]:
                raise PipelineError(f"{sample_id}: source.{field} is required")
        license_record = source.get("license")
        if (
            not isinstance(license_record, dict)
            or not license_record.get("name")
            or not license_record.get("url")
        ):
            raise PipelineError(f"{sample_id}: source.license is required")
        scope = sample.get("scope")
        if (
            not isinstance(scope, dict)
            or scope.get("in_scope") is not True
            or scope.get("ambiguity_flags") != []
        ):
            raise PipelineError(f"{sample_id}: sample is ambiguous or out of scope")
        if origin == "camera":
            capture = sample.get("capture")
            if not isinstance(capture, dict):
                raise PipelineError(f"{sample_id}: capture evidence is required")
            for field in ("camera_make", "camera_model", "captured_at"):
                if not capture.get(field):
                    raise PipelineError(f"{sample_id}: capture.{field} is required")
            if capture.get("edit_screen_status") != "pass":
                raise PipelineError(f"{sample_id}: camera edit screen must pass")
            if sample.get("generation") is not None:
                raise PipelineError(f"{sample_id}: camera generation must be null")
        else:
            generation = sample.get("generation")
            if not isinstance(generation, dict):
                raise PipelineError(f"{sample_id}: generation receipt is required")
            if generation.get("family_id") != expected.get("generator_family"):
                raise PipelineError(
                    f"{sample_id}: generator family differs from its slot"
                )
            for field in ("model_id", "provider", "settings"):
                if generation.get(field) in (None, ""):
                    raise PipelineError(f"{sample_id}: generation.{field} is required")
            if generation.get("input_image_used") is not False:
                raise PipelineError(
                    f"{sample_id}: only text-to-image outputs are in scope"
                )
            seed_status = generation.get("seed_status")
            seed = generation.get("seed")
            if seed_status == "recorded" and not isinstance(seed, int):
                raise PipelineError(f"{sample_id}: recorded seed must be an integer")
            if seed_status != "recorded" and seed is not None:
                raise PipelineError(f"{sample_id}: unavailable seed must be null")
            if sample.get("capture") is not None:
                raise PipelineError(f"{sample_id}: synthetic capture must be null")
        receipt = sample.get("provenance")
        receipt_path = sample.get("provenance_path")
        if not isinstance(receipt, dict) and not isinstance(receipt_path, str):
            raise PipelineError(
                f"{sample_id}: provenance or provenance_path is required"
            )
        input_path = sample_input_path(spec_path, sample)
        if not input_path.is_file():
            raise PipelineError(f"{sample_id}: input file does not exist: {input_path}")

    expected_slot_ids = set(slot_by_id)
    for group_id, slots in matrix.items():
        if set(slots) != expected_slot_ids:
            missing = sorted(expected_slot_ids - set(slots))
            extra = sorted(set(slots) - expected_slot_ids)
            raise PipelineError(
                f"{group_id}: incomplete slot matrix; missing={missing}, extra={extra}"
            )
        if len(group_prompts[group_id]) != 1:
            raise PipelineError(f"{group_id}: samples must share one frozen prompt")
        if len(group_categories[group_id]) != 1:
            raise PipelineError(f"{group_id}: samples must share one content category")
        if len(group_splits[group_id]) != 1:
            raise PipelineError(f"{group_id}: samples must share one split")
    return {
        "group_count": len(matrix),
        "sample_count": len(samples),
        "generator_families": sorted(families),
        "expected_slots": expected_slots,
        "target_group_count": target_group_count,
    }


def list_spec_groups(spec_path: Path) -> list[dict[str, Any]]:
    spec_path = spec_path.resolve()
    spec = read_json(spec_path)
    configured_groups = spec.get("groups")
    if configured_groups is not None:
        if not isinstance(configured_groups, list):
            raise PipelineError("build spec groups must be an array")
        expected_slots = spec.get("dataset", {}).get("expected_slots", [])
        planned_sample_count = (
            len(expected_slots) if isinstance(expected_slots, list) else 0
        )
        rows: list[dict[str, Any]] = []
        for group in configured_groups:
            if not isinstance(group, dict):
                raise PipelineError("every group must be an object")
            group_id = require_id(
                group.get("semantic_group_id"),
                "semantic_group_id",
            )
            samples = group.get("samples", [])
            if not isinstance(samples, list):
                raise PipelineError(f"{group_id}: samples must be an array")
            prompt = group.get("prompt")
            rows.append(
                {
                    "semantic_group_id": group_id,
                    "content_category": group.get("content_category", ""),
                    "prompt": (
                        prompt.get("text", "")
                        if isinstance(prompt, dict)
                        else ""
                    ),
                    "samples": samples,
                    "sample_count": len(samples) or planned_sample_count,
                    "sample_count_label": (
                        "defined" if samples else "on demand"
                    ),
                }
            )
        return sorted(rows, key=lambda row: row["semantic_group_id"])
    _, samples = load_spec(spec_path)
    groups: dict[str, dict[str, Any]] = {}
    for sample in samples:
        group_id = require_id(sample.get("semantic_group_id"), "semantic_group_id")
        prompt = sample.get("prompt")
        row = groups.setdefault(
            group_id,
            {
                "semantic_group_id": group_id,
                "content_category": sample.get("content_category", ""),
                "prompt": prompt.get("text", "") if isinstance(prompt, dict) else "",
                "samples": [],
            },
        )
        row["samples"].append({"sample_id": sample.get("sample_id", "")})
    return [groups[group_id] for group_id in sorted(groups)]


def validate_group_catalog(spec: dict[str, Any]) -> list[dict[str, Any]]:
    configured_groups = spec.get("groups")
    if not isinstance(configured_groups, list) or not configured_groups:
        raise PipelineError("build spec groups must be a non-empty array")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in configured_groups:
        if not isinstance(group, dict):
            raise PipelineError("every group must be an object")
        group_id = require_id(
            group.get("semantic_group_id"),
            "semantic_group_id",
        )
        if group_id in seen:
            raise PipelineError(f"duplicate semantic_group_id: {group_id}")
        seen.add(group_id)
        category = group.get("content_category")
        if not isinstance(category, str) or not category.strip():
            raise PipelineError(f"{group_id}: content_category is required")
        split = require_id(group.get("split"), f"{group_id}.split")
        prompt = group.get("prompt")
        if (
            not isinstance(prompt, dict)
            or not isinstance(prompt.get("text"), str)
            or not prompt["text"].strip()
            or prompt.get("frozen") is not True
        ):
            raise PipelineError(f"{group_id}: a frozen prompt is required")
        require_id(prompt.get("prompt_id"), f"{group_id}.prompt.prompt_id")
        samples = group.get("samples", [])
        if not isinstance(samples, list) or not all(
            isinstance(sample, dict) for sample in samples
        ):
            raise PipelineError(f"{group_id}: samples must be an array")
        result.append(
            {
                **group,
                "semantic_group_id": group_id,
                "content_category": category,
                "split": split,
                "prompt": prompt,
                "samples": samples,
            }
        )
    return result


def image_info(path: Path) -> dict[str, Any]:
    if shutil.which("magick") is None:
        raise PipelineError("ImageMagick 7 (`magick`) is required")
    result = subprocess.run(
        [
            "magick",
            "identify",
            "-format",
            "%w\t%h\t%m\t%[colorspace]",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise PipelineError(f"cannot inspect image {path}: {result.stderr.strip()}")
    parts = result.stdout.split("\t")
    if len(parts) != 4:
        raise PipelineError(f"unexpected ImageMagick output for {path}")
    return {
        "width": int(parts[0]),
        "height": int(parts[1]),
        "format": parts[2],
        "color_space": parts[3],
    }


def _png_info_stream(handle: Any, source: str) -> dict[str, Any]:
    chunks: list[str] = []
    if handle.read(8) != PNG_SIGNATURE:
        raise PipelineError(f"{source}: not a PNG")
    width = height = bit_depth = color_type = None
    while True:
        length_bytes = handle.read(4)
        if len(length_bytes) != 4:
            raise PipelineError(f"{source}: truncated PNG")
        length = struct.unpack(">I", length_bytes)[0]
        chunk_type = handle.read(4)
        data = handle.read(length)
        crc = handle.read(4)
        if len(chunk_type) != 4 or len(data) != length or len(crc) != 4:
            raise PipelineError(f"{source}: truncated PNG chunk")
        name = chunk_type.decode("ascii", errors="replace")
        chunks.append(name)
        if name == "IHDR":
            width, height, bit_depth, color_type = struct.unpack(
                ">IIBB", data[:10]
            )
        if name == "IEND":
            break
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "chunks": chunks,
    }


def png_info(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return _png_info_stream(handle, str(path))


def png_info_bytes(payload: bytes, source: str) -> dict[str, Any]:
    return _png_info_stream(io.BytesIO(payload), source)


def normalize(source: Path, target: Path, profile: dict[str, Any]) -> None:
    if shutil.which("magick") is None:
        raise PipelineError("ImageMagick 7 (`magick`) is required")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.png")
    command = [
        "magick",
        str(source),
        "-auto-orient",
        "-colorspace",
        "sRGB",
        "-resize",
        f"{profile['width']}x{profile['height']}^",
        "-gravity",
        "center",
        "-extent",
        f"{profile['width']}x{profile['height']}",
        "-strip",
        "-interlace",
        "none",
        "-define",
        "png:color-type=2",
        "-define",
        "png:bit-depth=8",
        str(temporary),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise PipelineError(f"normalization failed for {source}: {result.stderr.strip()}")
    os.replace(temporary, target)


def media_type(path: Path, image_format: str) -> str:
    guessed = mimetypes.guess_type(path.name)[0]
    if guessed and guessed.startswith("image/"):
        return guessed
    return f"image/{image_format.casefold()}"


def receipt_for(spec_path: Path, sample: dict[str, Any]) -> dict[str, Any]:
    if isinstance(sample.get("provenance"), dict):
        return sample["provenance"]
    value = sample["provenance_path"]
    path = Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else (spec_path.parent / path).resolve()
    return read_json(path)


def _parquet_symbols() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from datasets import Dataset, Features, Image, Sequence, Value
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise PipelineError(
            "Parquet dataset support is not installed. "
            "Install with `python3 -m pip install -e '.[parquet]'`."
        ) from error
    return Dataset, Features, Image, Sequence, Value, parquet


def parquet_features() -> Any:
    _, Features, Image, Sequence, Value, _ = _parquet_symbols()
    string = Value("string")
    integer = Value("int64")
    floating = Value("float64")
    boolean = Value("bool")
    return Features(
        {
            "schema_version": string,
            "sample_id": string,
            "semantic_group_id": string,
            "slot_id": string,
            "prompt_id": string,
            "prompt": string,
            "prompt_frozen": boolean,
            "origin_class": string,
            "content_category": string,
            "split": string,
            "source": {
                "collection_id": string,
                "source_record_id": string,
                "landing_page_url": string,
                "license": {
                    "name": string,
                    "url": string,
                },
                "details_json": string,
            },
            "capture": {
                "camera_make": string,
                "camera_model": string,
                "captured_at": string,
                "edit_screen_status": string,
                "software": string,
                "details_json": string,
            },
            "generation": {
                "family_id": string,
                "model_id": string,
                "model_revision": string,
                "provider": string,
                "settings": {
                    "width": integer,
                    "height": integer,
                    "num_inference_steps": integer,
                    "guidance_scale": floating,
                    "negative_prompt": string,
                    "max_sequence_length": integer,
                },
                "input_image_used": boolean,
                "seed_status": string,
                "seed": integer,
                "details_json": string,
            },
            "scope": {
                "in_scope": boolean,
                "ambiguity_flags": Sequence(string),
                "details_json": string,
            },
            "audit": {
                "selection_method": string,
                "automated_edit_screen": boolean,
                "cache_hit": boolean,
                "details_json": string,
            },
            "receipt_path": string,
            "receipt_sha256": string,
            "original_path": string,
            "original_sha256": string,
            "original_bytes": integer,
            "original_media_type": string,
            "original_width": integer,
            "original_height": integer,
            "image": Image(),
            "normalized_file_name": string,
            "normalized_sha256": string,
            "normalized_bytes": integer,
            "normalized_width": integer,
            "normalized_height": integer,
            "normalization_id": string,
            "data_file": string,
        }
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _typed_source(value: dict[str, Any]) -> dict[str, Any]:
    license_record = value.get("license")
    if not isinstance(license_record, dict):
        license_record = {}
    return {
        "collection_id": value.get("collection_id"),
        "source_record_id": value.get("source_record_id"),
        "landing_page_url": value.get("landing_page_url"),
        "license": {
            "name": license_record.get("name"),
            "url": license_record.get("url"),
        },
        "details_json": _canonical_json(value),
    }


def _typed_capture(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "camera_make": value.get("camera_make"),
        "camera_model": value.get("camera_model"),
        "captured_at": value.get("captured_at"),
        "edit_screen_status": value.get("edit_screen_status"),
        "software": value.get("software"),
        "details_json": _canonical_json(value),
    }


def _typed_generation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    settings = value.get("settings")
    if not isinstance(settings, dict):
        settings = {}
    return {
        "family_id": value.get("family_id"),
        "model_id": value.get("model_id"),
        "model_revision": value.get("model_revision"),
        "provider": value.get("provider"),
        "settings": {
            "width": settings.get("width"),
            "height": settings.get("height"),
            "num_inference_steps": settings.get("num_inference_steps"),
            "guidance_scale": settings.get("guidance_scale"),
            "negative_prompt": settings.get("negative_prompt"),
            "max_sequence_length": settings.get("max_sequence_length"),
        },
        "input_image_used": value.get("input_image_used"),
        "seed_status": value.get("seed_status"),
        "seed": value.get("seed"),
        "details_json": _canonical_json(value),
    }


def _typed_scope(value: dict[str, Any]) -> dict[str, Any]:
    flags = value.get("ambiguity_flags")
    if not isinstance(flags, list):
        flags = []
    return {
        "in_scope": value.get("in_scope"),
        "ambiguity_flags": flags,
        "details_json": _canonical_json(value),
    }


def _typed_audit(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    return {
        "selection_method": value.get("selection_method"),
        "automated_edit_screen": value.get("automated_edit_screen"),
        "cache_hit": value.get("cache_hit"),
        "details_json": _canonical_json(value),
    }


def _parquet_record(record: dict[str, Any]) -> dict[str, Any]:
    image = record.get("image")
    if not isinstance(image, dict) or not isinstance(image.get("bytes"), bytes):
        internal_path = record.get("_normalized_path")
        if not isinstance(internal_path, str):
            raise PipelineError(
                f"{record.get('sample_id')}: normalized image bytes are missing"
            )
        image = {
            "path": record["normalized_file_name"],
            "bytes": Path(internal_path).read_bytes(),
        }
    return {
        "schema_version": record["schema_version"],
        "sample_id": record["sample_id"],
        "semantic_group_id": record["semantic_group_id"],
        "slot_id": record["slot_id"],
        "prompt_id": record["prompt_id"],
        "prompt": record["prompt"],
        "prompt_frozen": record["prompt_frozen"],
        "origin_class": record["origin_class"],
        "content_category": record["content_category"],
        "split": record["split"],
        "source": _typed_source(record["source"]),
        "capture": _typed_capture(record.get("capture")),
        "generation": _typed_generation(record.get("generation")),
        "scope": _typed_scope(record["scope"]),
        "audit": _typed_audit(record.get("audit")),
        "receipt_path": record["receipt_path"],
        "receipt_sha256": record["receipt_sha256"],
        "original_path": record["original_path"],
        "original_sha256": record["original_sha256"],
        "original_bytes": record["original_bytes"],
        "original_media_type": record["original_media_type"],
        "original_width": record["original_width"],
        "original_height": record["original_height"],
        "image": image,
        "normalized_file_name": record["normalized_file_name"],
        "normalized_sha256": record["normalized_sha256"],
        "normalized_bytes": record["normalized_bytes"],
        "normalized_width": record["normalized_width"],
        "normalized_height": record["normalized_height"],
        "normalization_id": record["normalization_id"],
        "data_file": record["data_file"],
    }


def _hydrate_parquet_record(record: dict[str, Any]) -> dict[str, Any]:
    for field in ("source", "capture", "generation", "scope", "audit"):
        value = record.get(field)
        if not isinstance(value, dict):
            continue
        details = value.get("details_json")
        if not isinstance(details, str):
            continue
        try:
            decoded = json.loads(details)
        except json.JSONDecodeError as error:
            raise PipelineError(
                f"{record.get('sample_id')}: invalid {field}.details_json"
            ) from error
        if not isinstance(decoded, dict):
            raise PipelineError(
                f"{record.get('sample_id')}: {field}.details_json is not an object"
            )
        record[field] = decoded
    return record


def _partition_parquet_records(
    records: list[dict[str, Any]],
    target_bytes: int,
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current_group: list[dict[str, Any]] = []
    current_group_id: str | None = None
    for record in records:
        group_id = record["semantic_group_id"]
        if current_group and group_id != current_group_id:
            groups.append(current_group)
            current_group = []
        current_group_id = group_id
        current_group.append(record)
    if current_group:
        groups.append(current_group)

    shards: list[list[dict[str, Any]]] = []
    current_shard: list[dict[str, Any]] = []
    current_bytes = 0
    for group in groups:
        group_bytes = sum(int(record["normalized_bytes"]) for record in group)
        if current_shard and current_bytes + group_bytes > target_bytes:
            shards.append(current_shard)
            current_shard = []
            current_bytes = 0
        current_shard.extend(group)
        current_bytes += group_bytes
    if current_shard:
        shards.append(current_shard)
    return shards


def write_parquet_split(
    package: Path,
    split: str,
    records: list[dict[str, Any]],
    *,
    target_bytes: int = PARQUET_SHARD_TARGET_BYTES,
) -> list[dict[str, Any]]:
    if not records:
        return []
    Dataset, _, _, _, _, _ = _parquet_symbols()
    ordered = sorted(
        records,
        key=lambda item: (
            item["semantic_group_id"],
            item["slot_id"],
            item["sample_id"],
        ),
    )
    shards = _partition_parquet_records(ordered, target_bytes)
    entries: list[dict[str, Any]] = []
    for index, shard in enumerate(shards):
        relative = Path("data") / (
            f"{split}-{index:05d}-of-{len(shards):05d}.parquet"
        )
        for record in shard:
            record["data_file"] = relative.as_posix()
        rows = [_parquet_record(record) for record in shard]
        dataset = Dataset.from_list(rows, features=parquet_features())
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(
            target,
            batch_size=min(PARQUET_ROW_GROUP_SIZE, len(rows)),
        )
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256(target),
                "bytes": target.stat().st_size,
                "rows": len(rows),
            }
        )
    return entries


def read_parquet_records(path: Path) -> list[dict[str, Any]]:
    _, _, _, _, _, parquet = _parquet_symbols()
    try:
        table = parquet.read_table(path)
    except Exception as error:
        raise PipelineError(f"cannot read Parquet {path}: {error}") from error
    return [_hydrate_parquet_record(record) for record in table.to_pylist()]


def parquet_schema_errors(path: Path) -> list[str]:
    _, _, _, _, _, parquet = _parquet_symbols()
    try:
        schema = parquet.read_schema(path)
    except Exception as error:
        return [f"cannot read Parquet schema {path}: {error}"]
    expected_names = list(parquet_features())
    errors: list[str] = []
    if schema.names != expected_names:
        errors.append(f"{path}: Parquet columns differ from schema 3.0.0")
    metadata = schema.metadata or {}
    if b"huggingface" not in metadata:
        errors.append(f"{path}: Hugging Face feature metadata is missing")
    try:
        image_type = schema.field("image").type
        image_fields = {field.name: str(field.type) for field in image_type}
        if image_fields != {"bytes": "binary", "path": "string"}:
            errors.append(f"{path}: image is not a bytes/path struct")
        scope_type = schema.field("scope").type
        flags_type = scope_type.field("ambiguity_flags").type
        if str(flags_type.value_type) != "string":
            errors.append(f"{path}: ambiguity_flags is not list<string>")
    except (KeyError, TypeError, ValueError):
        errors.append(f"{path}: required nested Parquet fields are missing")
    return errors


def dataset_card(contract: dict[str, Any]) -> str:
    title = contract["title"]
    description = contract["description"]
    license_name = contract["license"]
    data_files = contract.get("files", {}).get("data", {})
    config_lines = [
        "configs:",
        "- config_name: default",
        "  data_files:",
    ]
    for split in sorted(data_files):
        config_lines.extend(
            [
                f"  - split: {split}",
                f"    path: data/{split}-*.parquet",
            ]
        )
    configs = "\n".join(config_lines)
    return f"""---
pretty_name: {json.dumps(title)}
license: other
task_categories:
- image-classification
tags:
- image
- image-forensics
- ai-generated-image-detection
- camera-provenance
{configs}
---

# {title}

{description}

This repository is produced by the MAI provenance-first dataset pipeline.
Normalized images and their typed metadata are embedded in Parquet; byte-identical
originals, provenance receipts, checksums, generator settings, and
semantic-group relationships are retained for audit.

Dataset-specific license identifier: `{license_name}`. Per-sample source licenses
in `data/*.parquet` govern the corresponding artifacts.

Load the normalized analysis images with:

```python
from datasets import load_dataset

dataset = load_dataset("OWNER/REPOSITORY", revision="PIN_A_COMMIT")
```

Pin a commit revision in every experiment. See `dataset.json`,
`groups.json`, and `validation_report.json` for the release contract and audit
summary.
"""


def build_package(
    spec_path: Path,
    output: Path,
    *,
    group_ids: list[str] | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    output = output.resolve()
    spec, samples = load_spec(spec_path)
    validate_dataset_design(spec)
    if isinstance(spec.get("groups"), list):
        group_catalog = validate_group_catalog(spec)
        available_group_ids = {
            group["semantic_group_id"] for group in group_catalog
        }
    else:
        group_catalog = []
        available_group_ids = {
            sample.get("semantic_group_id")
            for sample in samples
            if isinstance(sample.get("semantic_group_id"), str)
        }
    if group_ids:
        if len(group_ids) != len(set(group_ids)):
            raise PipelineError("semantic group selection contains duplicates")
        unknown = sorted(set(group_ids) - available_group_ids)
        if unknown:
            raise PipelineError("unknown semantic groups: " + ", ".join(unknown))
        selected = set(group_ids)
    else:
        selected = available_group_ids
    if group_catalog:
        selected_groups = [
            group
            for group in group_catalog
            if group["semantic_group_id"] in selected
        ]
        target_group_count = spec["dataset"]["target_group_count"]
        if not group_ids and len(selected_groups) < target_group_count:
            raise PipelineError(
                "full build is below dataset.target_group_count: "
                f"{len(selected_groups)} of {target_group_count} groups; "
                "select explicit --group-id values for a smoke build"
            )
        manual_group_ids = {
            group["semantic_group_id"]
            for group in selected_groups
            if group["samples"]
        }
        automatic_groups = [
            group for group in selected_groups if not group["samples"]
        ]
        samples = [
            sample
            for sample in samples
            if sample.get("semantic_group_id") in manual_group_ids
        ]
        if automatic_groups:
            from .preparation import preparation_plan, prepare_groups

            resolved_cache = (
                cache_dir.resolve()
                if cache_dir is not None
                else output.parent / "cache"
            )
            preparation = preparation_plan(
                spec,
                automatic_groups,
                resolved_cache,
            )
            if dry_run:
                return {
                    "status": "dry-run",
                    "group_count": len(selected_groups),
                    "sample_count": (
                        len(samples) + preparation["sample_count"]
                    ),
                    "selected_group_ids": sorted(selected),
                    "target_group_count": spec["dataset"]["target_group_count"],
                    "output": str(output),
                    "preparation": preparation,
                }
            prepared, preparation = prepare_groups(
                spec,
                automatic_groups,
                resolved_cache,
            )
            samples.extend(prepared)
        else:
            preparation = {
                "group_count": 0,
                "sample_count": 0,
                "cache_hits": 0,
                "camera_downloads": 0,
                "generation_jobs": 0,
                "configured_credentials": [],
                "required_credentials": [],
                "missing_credentials": [],
            }
    else:
        selected_groups = []
        preparation = None
        if group_ids:
            samples = [
                sample
                for sample in samples
                if sample.get("semantic_group_id") in selected
            ]
    summary = validate_spec(spec_path, spec, samples)
    summary["selected_group_ids"] = sorted(
        {
            sample["semantic_group_id"]
            for sample in samples
        }
    )
    if not group_ids and summary["group_count"] < summary["target_group_count"]:
        raise PipelineError(
            "full build is below dataset.target_group_count: "
            f"{summary['group_count']} of {summary['target_group_count']} groups; "
            "select explicit --group-id values for a smoke build"
        )
    if preparation is not None:
        summary["preparation"] = preparation
    if dry_run:
        return {"status": "dry-run", **summary, "output": str(output)}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not force:
            raise PipelineError(f"output already exists: {output}; use --force")
        marker = output / "dataset.json"
        if not marker.is_file():
            raise PipelineError(
                f"refusing to replace unrecognized directory without dataset.json: {output}"
            )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-build-", dir=output.parent)
    )
    try:
        dataset_spec = spec["dataset"]
        profile = {**NORMALIZATION, **dataset_spec.get("normalization", {})}
        contract = {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "dataset_id": dataset_spec["dataset_id"],
            "title": dataset_spec["title"],
            "description": dataset_spec["description"],
            "license": dataset_spec["license"],
            "created_at": utc_now(),
            "builder": {"name": "mai-research", "version": __version__},
            "build_spec": {
                "schema_version": spec["schema_version"],
                "sha256": sha256(spec_path),
            },
            "selection": {
                "method": "explicit" if group_ids else "all",
                "semantic_groups": summary["selected_group_ids"],
                "sample_count": summary["sample_count"],
            },
            "normalization": profile,
            "design": {
                "expected_slots": dataset_spec["expected_slots"],
                "target_group_count": dataset_spec["target_group_count"],
                "group_locked_splits": True,
                "originals_are_immutable": True,
                "analysis_unit": "normalized",
                "seed_base": dataset_spec.get("seed_base"),
                "prompt_policy": dataset_spec.get("prompt_policy"),
                "generation_review": dataset_spec.get("generation_review"),
                "camera_acquisition": dataset_spec.get("camera_acquisition"),
                "generators": dataset_spec.get("generators"),
            },
            "files": {
                "group_index": "groups.json",
                "validation_report": "validation_report.json",
                "data": {},
            },
        }
        records_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for index, sample in enumerate(
            sorted(
                samples,
                key=lambda item: (
                    item["semantic_group_id"],
                    item["slot_id"],
                    item["sample_id"],
                ),
            ),
            1,
        ):
            sample_id = sample["sample_id"]
            group_id = sample["semantic_group_id"]
            split = sample.get("split", "train")
            original_source = sample_input_path(spec_path, sample)
            suffix = original_source.suffix.casefold() or ".bin"
            family = (
                sample.get("generation", {}).get("family_id", "camera")
                if sample["origin_class"] == "synthetic"
                else "camera"
            )
            original_relative = (
                Path("originals")
                / sample["origin_class"]
                / family
                / f"{sample_id}{suffix}"
            )
            original_target = staging / original_relative
            original_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original_source, original_target)
            original_info = image_info(original_target)

            normalized_relative = (
                Path(".normalized") / split / f"{sample_id}.png"
            )
            normalized_target = staging / normalized_relative
            normalize(original_target, normalized_target, profile)
            normalized_info = png_info(normalized_target)

            receipt_relative = Path("receipts") / f"{sample_id}.json"
            receipt = receipt_for(spec_path, sample)
            write_json(staging / receipt_relative, receipt)

            record = {
                "schema_version": PACKAGE_SCHEMA_VERSION,
                "sample_id": sample_id,
                "semantic_group_id": group_id,
                "slot_id": sample["slot_id"],
                "prompt_id": sample["prompt"]["prompt_id"],
                "prompt": sample["prompt"]["text"],
                "prompt_frozen": True,
                "origin_class": sample["origin_class"],
                "content_category": sample["content_category"],
                "split": split,
                "source": sample["source"],
                "capture": sample.get("capture"),
                "generation": sample.get("generation"),
                "scope": sample["scope"],
                "audit": sample.get("audit", {}),
                "receipt_path": receipt_relative.as_posix(),
                "receipt_sha256": sha256(staging / receipt_relative),
                "original_path": original_relative.as_posix(),
                "original_sha256": sha256(original_target),
                "original_bytes": original_target.stat().st_size,
                "original_media_type": media_type(
                    original_target, original_info["format"]
                ),
                "original_width": original_info["width"],
                "original_height": original_info["height"],
                "normalized_file_name": f"{sample_id}.png",
                "normalized_sha256": sha256(normalized_target),
                "normalized_bytes": normalized_target.stat().st_size,
                "normalized_width": normalized_info["width"],
                "normalized_height": normalized_info["height"],
                "normalization_id": profile["transformation_id"],
                "_normalized_path": str(normalized_target),
            }
            records_by_split[split].append(record)
            print(f"[{index}/{len(samples)}] {sample_id}")

        for split, records in sorted(records_by_split.items()):
            contract["files"]["data"][split] = write_parquet_split(
                staging,
                split,
                records,
            )
        shutil.rmtree(staging / ".normalized")
        all_records = [
            record
            for records in records_by_split.values()
            for record in records
        ]
        groups = group_index_from_records(
            contract["dataset_id"],
            all_records,
            schema_version=PACKAGE_SCHEMA_VERSION,
        )
        write_json(staging / "dataset.json", contract)
        write_json(staging / "groups.json", groups)
        (staging / "README.md").write_text(dataset_card(contract), encoding="utf-8")

        status, report = validate_package(staging)
        write_json(staging / "validation_report.json", report)
        if status:
            raise PipelineError(
                "built package failed validation: " + "; ".join(report["errors"][:5])
            )
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
        return {**report, "output": str(output)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_package_records(
    package: Path,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    schema_version = contract.get("schema_version")
    if schema_version == PACKAGE_SCHEMA_VERSION:
        data = contract.get("files", {}).get("data", {})
        if not isinstance(data, dict) or not data:
            raise PipelineError("dataset contract has no Parquet data files")
        for split, entries in data.items():
            if not isinstance(entries, list) or not entries:
                raise PipelineError(f"dataset split {split} has no Parquet shards")
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(
                    entry.get("path"), str
                ):
                    raise PipelineError(
                        f"dataset split {split} has an invalid Parquet shard"
                    )
                records.extend(
                    read_parquet_records(safe_path(package, entry["path"]))
                )
        return records
    if schema_version != LEGACY_PACKAGE_SCHEMA_VERSION:
        raise PipelineError(
            f"unsupported dataset schema_version: {schema_version!r}"
        )
    metadata = contract.get("files", {}).get("metadata", {})
    if not isinstance(metadata, dict) or not metadata:
        raise PipelineError("dataset contract has no metadata files")
    for relative in metadata.values():
        if not isinstance(relative, str):
            raise PipelineError("dataset metadata path must be a string")
        records.extend(read_jsonl(safe_path(package, relative)))
    return records


def group_index_from_records(
    dataset_id: str,
    records: list[dict[str, Any]],
    *,
    schema_version: str = PACKAGE_SCHEMA_VERSION,
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for record in sorted(
        records,
        key=lambda item: (
            item["semantic_group_id"],
            item["slot_id"],
            item["sample_id"],
        ),
    ):
        group_id = record["semantic_group_id"]
        group = groups.setdefault(
            group_id,
            {
                "semantic_group_id": group_id,
                "prompt_id": record["prompt_id"],
                "prompt": record["prompt"],
                "content_category": record["content_category"],
                "split": record["split"],
                **(
                    {"data_file": record["data_file"]}
                    if schema_version == PACKAGE_SCHEMA_VERSION
                    else {}
                ),
                "samples": [],
            },
        )
        sample = {
            "sample_id": record["sample_id"],
            "slot_id": record["slot_id"],
            "origin_class": record["origin_class"],
            "generator_family": (
                record["generation"].get("family_id")
                if isinstance(record.get("generation"), dict)
                else None
            ),
            "original_path": record["original_path"],
        }
        if schema_version == PACKAGE_SCHEMA_VERSION:
            sample["normalized_file_name"] = record["normalized_file_name"]
        else:
            sample["normalized_path"] = record["normalized_path"]
        group["samples"].append(sample)
    return {
        "schema_version": schema_version,
        "dataset_id": dataset_id,
        "groups": [groups[key] for key in sorted(groups)],
    }


def validate_package(package: Path) -> tuple[int, dict[str, Any]]:
    package = package.resolve()
    try:
        contract = read_json(package / "dataset.json")
    except PipelineError as error:
        validation = Validation()
        report = validation.report(
            group_count=0,
            sample_count=0,
        )
        report["status"] = "fail"
        report["errors"] = [str(error)]
        return 1, report
    package_schema = contract.get("schema_version")
    validation = Validation(
        package_schema
        if isinstance(package_schema, str)
        else PACKAGE_SCHEMA_VERSION
    )
    validation.require(
        package_schema in SUPPORTED_PACKAGE_SCHEMA_VERSIONS,
        "unsupported dataset schema_version",
    )
    try:
        records = load_package_records(package, contract)
    except PipelineError as error:
        report = validation.report(
            group_count=0,
            sample_count=0,
        )
        report["status"] = "fail"
        report["errors"].append(str(error))
        return 1, report

    parquet_paths: set[str] = set()
    parquet_splits: dict[str, str] = {}
    if package_schema == PACKAGE_SCHEMA_VERSION:
        data = contract.get("files", {}).get("data", {})
        if isinstance(data, dict):
            for split, entries in data.items():
                if not isinstance(entries, list):
                    validation.require(
                        False,
                        f"{split}: Parquet shard list is invalid",
                    )
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        validation.require(
                            False,
                            f"{split}: Parquet shard entry is invalid",
                        )
                        continue
                    relative = entry.get("path")
                    validation.require(
                        isinstance(relative, str) and bool(relative),
                        f"{split}: Parquet shard path is missing",
                    )
                    if not isinstance(relative, str):
                        continue
                    validation.require(
                        relative not in parquet_paths,
                        f"Parquet shard path reused: {relative}",
                    )
                    parquet_paths.add(relative)
                    parquet_splits[relative] = split
                    try:
                        path = safe_path(package, relative)
                    except PipelineError as error:
                        validation.require(False, str(error))
                        continue
                    validation.require(
                        path.is_file(),
                        f"missing Parquet shard: {relative}",
                    )
                    if not path.is_file():
                        continue
                    validation.require(
                        path.stat().st_size == entry.get("bytes"),
                        f"{relative}: byte count mismatch",
                    )
                    validation.require(
                        sha256(path) == entry.get("sha256"),
                        f"{relative}: checksum mismatch",
                    )
                    for error in parquet_schema_errors(path):
                        validation.require(False, error)
                    observed_rows = sum(
                        record.get("data_file") == relative
                        for record in records
                    )
                    validation.require(
                        observed_rows == entry.get("rows"),
                        f"{relative}: row count mismatch",
                    )
    expected_slots = contract.get("design", {}).get("expected_slots", [])
    expected_slot_ids = {
        slot.get("slot_id") for slot in expected_slots if isinstance(slot, dict)
    }
    slot_by_id = {
        slot.get("slot_id"): slot for slot in expected_slots if isinstance(slot, dict)
    }
    validation.require(bool(expected_slot_ids), "dataset has no expected slots")
    ids: set[str] = set()
    paths: set[str] = set()
    image_names: set[str] = set()
    hashes: set[str] = set()
    group_slots: dict[str, set[str]] = defaultdict(set)
    group_prompts: dict[str, set[tuple[str, str]]] = defaultdict(set)
    group_categories: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    group_data_files: dict[str, set[str]] = defaultdict(set)
    families: set[str] = set()
    review_design = contract.get("design", {}).get("generation_review", {})
    review_required = (
        isinstance(review_design, dict)
        and review_design.get("require_explicit_decision") is True
    )
    review_method = (
        review_design.get("selection_method")
        if isinstance(review_design, dict)
        else None
    )
    for record in records:
        sample_id = record.get("sample_id", "<missing>")
        validation.require(
            record.get("schema_version") == package_schema,
            f"{sample_id}: unsupported schema_version",
        )
        validation.require(sample_id not in ids, f"duplicate sample_id: {sample_id}")
        ids.add(sample_id)
        group_id = record.get("semantic_group_id")
        slot_id = record.get("slot_id")
        validation.require(
            slot_id in expected_slot_ids,
            f"{sample_id}: unexpected slot_id {slot_id}",
        )
        validation.require(
            slot_id not in group_slots[group_id],
            f"{group_id}: duplicate slot {slot_id}",
        )
        group_slots[group_id].add(slot_id)
        group_prompts[group_id].add((record.get("prompt_id"), record.get("prompt")))
        group_categories[group_id].add(record.get("content_category"))
        group_splits[group_id].add(record.get("split"))
        origin = record.get("origin_class")
        validation.require(origin in {"camera", "synthetic"}, f"{sample_id}: bad origin")
        expected = slot_by_id.get(slot_id, {})
        validation.require(
            origin == expected.get("origin_class"),
            f"{sample_id}: origin differs from slot",
        )
        if origin == "camera":
            capture = record.get("capture")
            validation.require(
                isinstance(capture, dict)
                and capture.get("edit_screen_status") == "pass",
                f"{sample_id}: camera evidence did not pass",
            )
            validation.require(
                record.get("generation") is None,
                f"{sample_id}: camera generation must be null",
            )
        elif origin == "synthetic":
            generation = record.get("generation")
            validation.require(
                isinstance(generation, dict)
                and generation.get("input_image_used") is False,
                f"{sample_id}: synthetic generation is incomplete or hybrid",
            )
            if isinstance(generation, dict):
                family = generation.get("family_id")
                families.add(family)
                validation.require(
                    family == expected.get("generator_family"),
                    f"{sample_id}: generator family differs from slot",
                )
                if review_required:
                    configured_candidates = review_design.get(
                        "candidates_per_slot"
                    )
                    validation.require(
                        isinstance(generation.get("candidate_index"), int)
                        and isinstance(generation.get("candidate_count"), int),
                        f"{sample_id}: reviewed candidate metadata is missing",
                    )
                    validation.require(
                        generation.get("candidate_count")
                        == configured_candidates
                        and 0
                        <= generation.get("candidate_index", -1)
                        < generation.get("candidate_count", 0),
                        f"{sample_id}: reviewed candidate metadata differs "
                        "from contract",
                    )
            if review_required:
                audit = record.get("audit")
                validation.require(
                    isinstance(audit, dict)
                    and audit.get("review_status") == "accepted",
                    f"{sample_id}: explicit generation review is missing",
                )
                validation.require(
                    isinstance(audit, dict)
                    and audit.get("selection_method") == review_method,
                    f"{sample_id}: generation review method differs from contract",
                )
                validation.require(
                    isinstance(audit, dict)
                    and isinstance(audit.get("reviewer"), str)
                    and bool(audit["reviewer"])
                    and isinstance(audit.get("reviewed_at"), str)
                    and bool(audit["reviewed_at"]),
                    f"{sample_id}: generation reviewer metadata is missing",
                )
        scope = record.get("scope")
        validation.require(
            isinstance(scope, dict)
            and scope.get("in_scope") is True
            and scope.get("ambiguity_flags") == [],
            f"{sample_id}: ambiguous or out of scope",
        )
        file_roles = ["receipt_path", "original_path"]
        if package_schema == LEGACY_PACKAGE_SCHEMA_VERSION:
            file_roles.append("normalized_path")
        for role in file_roles:
            relative = record.get(role)
            validation.require(
                isinstance(relative, str) and bool(relative),
                f"{sample_id}: {role} is missing",
            )
            if not isinstance(relative, str):
                continue
            try:
                path = safe_path(package, relative)
            except PipelineError as error:
                validation.require(False, f"{sample_id}: {error}")
                continue
            validation.require(path.is_file(), f"{sample_id}: missing {relative}")
            validation.require(
                relative not in paths,
                f"artifact path reused: {relative}",
            )
            paths.add(relative)
            if not path.is_file():
                continue
            if role == "receipt_path":
                expected_hash = record.get("receipt_sha256")
            elif role == "original_path":
                expected_hash = record.get("original_sha256")
                validation.require(
                    path.stat().st_size == record.get("original_bytes"),
                    f"{sample_id}: original byte count mismatch",
                )
            else:
                expected_hash = record.get("normalized_sha256")
                validation.require(
                    path.stat().st_size == record.get("normalized_bytes"),
                    f"{sample_id}: normalized byte count mismatch",
                )
            observed_hash = sha256(path)
            validation.require(
                observed_hash == expected_hash,
                f"{sample_id}: {role} checksum mismatch",
            )
            if role != "receipt_path":
                validation.require(
                    observed_hash not in hashes,
                    f"image content reused: {observed_hash}",
                )
                hashes.add(observed_hash)
            if role == "normalized_path":
                try:
                    info = png_info(path)
                except PipelineError as error:
                    validation.require(False, str(error))
                else:
                    profile = contract.get("normalization", {})
                    validation.require(
                        info["width"] == profile.get("width")
                        and info["height"] == profile.get("height"),
                        f"{sample_id}: normalized dimensions differ from profile",
                    )
                    validation.require(
                        info["bit_depth"] == 8 and info["color_type"] == 2,
                        f"{sample_id}: normalized PNG is not 8-bit RGB",
                    )
                    leaking = {"eXIf", "tEXt", "zTXt", "iTXt", "tIME"} & set(
                        info["chunks"]
                    )
                    validation.require(
                        not leaking,
                        f"{sample_id}: normalized metadata remains: {sorted(leaking)}",
                    )
        if package_schema == PACKAGE_SCHEMA_VERSION:
            image = record.get("image")
            payload = image.get("bytes") if isinstance(image, dict) else None
            logical_path = image.get("path") if isinstance(image, dict) else None
            validation.require(
                isinstance(payload, bytes),
                f"{sample_id}: embedded image bytes are missing",
            )
            validation.require(
                isinstance(logical_path, str)
                and logical_path == record.get("normalized_file_name"),
                f"{sample_id}: embedded image filename mismatch",
            )
            if isinstance(logical_path, str):
                validation.require(
                    logical_path == f"{sample_id}.png",
                    f"{sample_id}: embedded image filename is not deterministic",
                )
                validation.require(
                    logical_path not in image_names,
                    f"embedded image filename reused: {logical_path}",
                )
                image_names.add(logical_path)
            validation.require(
                record.get("data_file") in parquet_paths,
                f"{sample_id}: data_file is not declared",
            )
            validation.require(
                parquet_splits.get(record.get("data_file")) == record.get("split"),
                f"{sample_id}: data_file belongs to a different split",
            )
            if isinstance(record.get("data_file"), str):
                group_data_files[group_id].add(record["data_file"])
            if isinstance(payload, bytes):
                validation.require(
                    len(payload) == record.get("normalized_bytes"),
                    f"{sample_id}: normalized byte count mismatch",
                )
                observed_hash = hashlib.sha256(payload).hexdigest()
                validation.require(
                    observed_hash == record.get("normalized_sha256"),
                    f"{sample_id}: embedded image checksum mismatch",
                )
                validation.require(
                    observed_hash not in hashes,
                    f"image content reused: {observed_hash}",
                )
                hashes.add(observed_hash)
                try:
                    info = png_info_bytes(payload, f"{sample_id}:image")
                except PipelineError as error:
                    validation.require(False, str(error))
                else:
                    profile = contract.get("normalization", {})
                    validation.require(
                        info["width"] == profile.get("width")
                        and info["height"] == profile.get("height"),
                        f"{sample_id}: normalized dimensions differ from profile",
                    )
                    validation.require(
                        info["bit_depth"] == 8 and info["color_type"] == 2,
                        f"{sample_id}: normalized PNG is not 8-bit RGB",
                    )
                    leaking = {"eXIf", "tEXt", "zTXt", "iTXt", "tIME"} & set(
                        info["chunks"]
                    )
                    validation.require(
                        not leaking,
                        f"{sample_id}: normalized metadata remains: {sorted(leaking)}",
                    )
    for group_id in sorted(group_slots):
        validation.require(
            group_slots[group_id] == expected_slot_ids,
            f"{group_id}: incomplete slot matrix",
        )
        validation.require(
            len(group_prompts[group_id]) == 1,
            f"{group_id}: prompt mismatch",
        )
        validation.require(
            len(group_categories[group_id]) == 1,
            f"{group_id}: content category mismatch",
        )
        validation.require(
            len(group_splits[group_id]) == 1,
            f"{group_id}: split leakage",
        )
        if package_schema == PACKAGE_SCHEMA_VERSION:
            validation.require(
                len(group_data_files[group_id]) == 1,
                f"{group_id}: semantic group spans Parquet shards",
            )
    validation.require(
        len(families) >= 2,
        "dataset contains fewer than two generator families",
    )
    selection = contract.get("selection", {})
    target_group_count = contract.get("design", {}).get("target_group_count")
    validation.require(
        isinstance(target_group_count, int) and target_group_count > 0,
        "dataset has no valid target_group_count",
    )
    if (
        isinstance(selection, dict)
        and selection.get("method") == "all"
        and isinstance(target_group_count, int)
    ):
        validation.require(
            len(group_slots) >= target_group_count,
            "full package is below target_group_count",
        )
    expected_index = group_index_from_records(
        contract.get("dataset_id"),
        records,
        schema_version=package_schema,
    )
    try:
        observed_index = read_json(package / "groups.json")
    except PipelineError as error:
        validation.require(False, str(error))
    else:
        validation.require(
            observed_index == expected_index,
            "groups.json differs from sample metadata",
        )
    validation.require((package / "README.md").is_file(), "dataset card is missing")
    report = validation.report(
        dataset_id=contract.get("dataset_id"),
        group_count=len(group_slots),
        sample_count=len(records),
        generator_families=sorted(family for family in families if family),
    )
    return (0 if report["status"] == "pass" else 1), report


def initialize_spec(path: Path, *, force: bool = False) -> None:
    path = path.resolve()
    if path.exists() and not force:
        raise PipelineError(f"spec already exists: {path}; use --force")
    template = {
        "schema_version": BUILD_SPEC_SCHEMA_VERSION,
        "dataset": {
            "dataset_id": "mai-v1",
            "title": "MAI camera-origin versus generated image atlas v1",
            "description": "A provenance-controlled embedding-atlas dataset.",
            "license": "mixed-per-sample",
            "target_group_count": 200,
            "seed_base": 1729,
            "prompt_policy": {
                "template_id": "natural-photo-v2",
                "template": (
                    "A natural photograph depicting {concept}. Realistic "
                    "lighting, materials, textures, and ordinary photographic "
                    "composition."
                ),
            },
            "generation_review": {
                "candidates_per_slot": 1,
                "require_explicit_decision": False,
                "selection_method": "first-candidate-v1",
            },
            "camera_acquisition": {
                "adapter": "wikimedia-commons",
                "api_url": "https://commons.wikimedia.org/w/api.php",
                "search_limit": 40,
                "candidate_limit": 8,
                "require_camera_exif": True,
                "reject_detected_editors": True,
            },
            "generators": {
                "flux": {
                    "family_id": "flux",
                    "adapter": "local-diffusers",
                    "model_id": "black-forest-labs/FLUX.1-schnell",
                    "device": "auto",
                    "settings": {
                        "width": 1024,
                        "height": 1024,
                        "num_inference_steps": 4,
                        "guidance_scale": 0.0,
                        "max_sequence_length": 256,
                    },
                    "output_terms_url": (
                        "https://huggingface.co/black-forest-labs/"
                        "FLUX.1-schnell/blob/main/LICENSE.md"
                    ),
                },
                "stable-diffusion-xl": {
                    "family_id": "stable-diffusion-xl",
                    "adapter": "local-diffusers",
                    "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                    "device": "auto",
                    "settings": {
                        "width": 1024,
                        "height": 1024,
                        "num_inference_steps": 30,
                        "guidance_scale": 6.0,
                        "negative_prompt": (
                            "illustration, drawing, painting, cartoon, CGI, "
                            "3D render, graphic design, collage, border, frame, "
                            "watermark"
                        ),
                    },
                    "output_terms_url": (
                        "https://huggingface.co/stabilityai/"
                        "stable-diffusion-xl-base-1.0/blob/main/LICENSE.md"
                    ),
                },
                "stable-diffusion-1-5": {
                    "family_id": "stable-diffusion-1-5",
                    "adapter": "local-diffusers",
                    "model_id": (
                        "stable-diffusion-v1-5/stable-diffusion-v1-5"
                    ),
                    "device": "auto",
                    "settings": {
                        "width": 512,
                        "height": 512,
                        "num_inference_steps": 30,
                        "guidance_scale": 7.0,
                        "negative_prompt": (
                            "illustration, drawing, painting, cartoon, CGI, "
                            "3D render, graphic design, collage, border, frame, "
                            "watermark"
                        ),
                    },
                    "output_terms_url": (
                        "https://huggingface.co/stable-diffusion-v1-5/"
                        "stable-diffusion-v1-5/blob/main/LICENSE.md"
                    ),
                },
            },
            "expected_slots": [
                {"slot_id": "camera", "origin_class": "camera"},
                {
                    "slot_id": "flux-replicate-0",
                    "origin_class": "synthetic",
                    "generator_family": "flux",
                },
                {
                    "slot_id": "flux-replicate-1",
                    "origin_class": "synthetic",
                    "generator_family": "flux",
                },
                {
                    "slot_id": "stable-diffusion-xl-replicate-0",
                    "origin_class": "synthetic",
                    "generator_family": "stable-diffusion-xl",
                },
                {
                    "slot_id": "stable-diffusion-xl-replicate-1",
                    "origin_class": "synthetic",
                    "generator_family": "stable-diffusion-xl",
                },
                {
                    "slot_id": "stable-diffusion-1-5-replicate-0",
                    "origin_class": "synthetic",
                    "generator_family": "stable-diffusion-1-5",
                },
                {
                    "slot_id": "stable-diffusion-1-5-replicate-1",
                    "origin_class": "synthetic",
                    "generator_family": "stable-diffusion-1-5",
                },
            ],
        },
        "groups": [],
    }
    write_json(path, template)
