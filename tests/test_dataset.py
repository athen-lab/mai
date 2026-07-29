from __future__ import annotations

import binascii
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

from mai.dataset import (
    PipelineError,
    build_package,
    group_index_from_records,
    list_spec_groups,
    read_parquet_records,
    read_json,
    validate_package,
    write_json,
    write_jsonl,
)
from mai.hub import list_remote_groups, publish_package, pull_groups


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind)
    checksum = binascii.crc32(payload, checksum)
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", checksum & 0xFFFFFFFF)
    )


def make_png(path: Path, color: tuple[int, int, int]) -> None:
    width = height = 8
    row = b"\x00" + bytes(color) * width
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + png_chunk(b"IDAT", zlib.compress(row * height))
        + png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_fixture(root: Path, group_count: int = 2) -> Path:
    slots = [
        {"slot_id": "camera", "origin_class": "camera"},
        {
            "slot_id": "flux-seed-0",
            "origin_class": "synthetic",
            "generator_family": "flux",
        },
        {
            "slot_id": "sdxl-seed-0",
            "origin_class": "synthetic",
            "generator_family": "sdxl",
        },
    ]
    samples: list[dict[str, object]] = []
    colors = iter(
        [
            (220, 20, 60),
            (40, 110, 210),
            (30, 180, 100),
            (240, 150, 30),
            (130, 50, 200),
            (20, 190, 210),
        ]
    )
    for group_number in range(1, group_count + 1):
        group_id = f"group-{group_number:03d}"
        prompt = {
            "prompt_id": f"prompt-{group_number:03d}",
            "text": f"A camera photograph of controlled subject {group_number}",
            "frozen": True,
        }
        for slot_number, slot in enumerate(slots):
            sample_id = f"{group_id}-{slot['slot_id']}"
            image = root / "inputs" / f"{sample_id}.png"
            make_png(image, next(colors))
            origin = slot["origin_class"]
            generation = None
            capture = None
            if origin == "camera":
                capture = {
                    "camera_make": "Test Camera Company",
                    "camera_model": "AuditCam 1",
                    "captured_at": "2025-01-02T03:04:05Z",
                    "edit_screen_status": "pass",
                }
            else:
                generation = {
                    "family_id": slot["generator_family"],
                    "model_id": f"test/{slot['generator_family']}",
                    "model_revision": "0123456789abcdef",
                    "provider": "local-test",
                    "settings": {
                        "steps": 20,
                        "guidance": 5.0,
                        "width": 1024,
                        "height": 1024,
                    },
                    "input_image_used": False,
                    "seed_status": "recorded",
                    "seed": group_number * 100 + slot_number,
                }
            samples.append(
                {
                    "sample_id": sample_id,
                    "semantic_group_id": group_id,
                    "slot_id": slot["slot_id"],
                    "origin_class": origin,
                    "content_category": "controlled-object",
                    "split": "train",
                    "prompt": prompt,
                    "input_path": str(image.relative_to(root)),
                    "source": {
                        "collection_id": (
                            "test-camera-archive"
                            if origin == "camera"
                            else f"test-{slot['generator_family']}-run"
                        ),
                        "source_record_id": sample_id,
                        "landing_page_url": f"https://example.test/{sample_id}",
                        "license": {
                            "name": "CC-BY-4.0",
                            "url": "https://creativecommons.org/licenses/by/4.0/",
                        },
                    },
                    "scope": {"in_scope": True, "ambiguity_flags": []},
                    "provenance": {
                        "kind": "test-receipt",
                        "sample_id": sample_id,
                        "verified": True,
                    },
                    "capture": capture,
                    "generation": generation,
                }
            )
    spec = {
        "schema_version": "2.0.0",
        "dataset": {
            "dataset_id": "mai-test-v1",
            "title": "MAI test dataset",
            "description": "Generated test fixture for the package pipeline.",
            "license": "mixed-per-sample",
            "target_group_count": group_count,
            "expected_slots": slots,
        },
        "samples": samples,
    }
    spec_path = root / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path


@unittest.skipUnless(shutil.which("magick"), "ImageMagick 7 is required")
class DatasetPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.spec = write_fixture(self.root)
        self.package = self.root / "package"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dry_run_only_summarizes(self) -> None:
        report = build_package(self.spec, self.package, dry_run=True)
        self.assertEqual(report["status"], "dry-run")
        self.assertEqual(report["group_count"], 2)
        self.assertEqual(report["sample_count"], 6)
        self.assertFalse(self.package.exists())

    def test_build_creates_valid_hugging_face_package(self) -> None:
        report = build_package(self.spec, self.package)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["group_count"], 2)
        self.assertEqual(report["sample_count"], 6)
        parquet_files = sorted((self.package / "data").glob("train-*.parquet"))
        self.assertEqual(len(parquet_files), 1)
        self.assertFalse((self.package / "data/train/images").exists())
        self.assertFalse((self.package / "data/train/metadata.jsonl").exists())
        self.assertTrue((self.package / "originals/camera/camera").is_dir())
        self.assertFalse((self.package / "manifest.jsonl").exists())
        self.assertFalse((self.package / "acquisition.jsonl").exists())
        contract = read_json(self.package / "dataset.json")
        self.assertEqual(contract["schema_version"], "3.0.0")
        self.assertEqual(contract["files"]["data"]["train"][0]["rows"], 6)

        from datasets import Dataset, Image, load_dataset

        dataset = Dataset.from_parquet(
            str(parquet_files[0]),
            cache_dir=str(self.root / "datasets-cache"),
        )
        self.assertIsInstance(dataset.features["image"], Image)
        undecoded = dataset.cast_column("image", Image(decode=False))
        image = undecoded[0]["image"]
        self.assertIsInstance(image["bytes"], bytes)
        self.assertTrue(image["path"].endswith(".png"))
        self.assertEqual(
            undecoded.features["scope"]["ambiguity_flags"].feature.dtype,
            "string",
        )
        loaded = load_dataset(
            str(self.package),
            cache_dir=str(self.root / "hub-layout-cache"),
        )
        self.assertEqual(loaded.num_rows["train"], 6)
        self.assertIsInstance(loaded["train"].features["image"], Image)
        self.assertEqual(loaded["train"][0]["image"].size, (512, 512))
        status, validation = validate_package(self.package)
        self.assertEqual(status, 0, validation)

    def test_explicit_review_contract_requires_accepted_candidates(self) -> None:
        spec = read_json(self.spec)
        spec["dataset"]["generation_review"] = {
            "candidates_per_slot": 2,
            "require_explicit_decision": True,
            "selection_method": "explicit-first-passing-v1",
        }
        write_json(self.spec, spec)
        with self.assertRaisesRegex(
            PipelineError,
            "explicit generation review is missing",
        ):
            build_package(self.spec, self.package)

    def test_explicit_review_metadata_builds_valid_package(self) -> None:
        spec = read_json(self.spec)
        spec["dataset"]["generation_review"] = {
            "candidates_per_slot": 2,
            "require_explicit_decision": True,
            "selection_method": "explicit-first-passing-v1",
        }
        for sample in spec["samples"]:
            if sample["origin_class"] != "synthetic":
                continue
            sample["generation"]["candidate_index"] = 0
            sample["generation"]["candidate_count"] = 2
            sample["audit"] = {
                "selection_method": "explicit-first-passing-v1",
                "review_status": "accepted",
                "candidate_index": 0,
                "candidate_count": 2,
                "reviewer": "test-reviewer",
                "reviewed_at": "2026-01-01T00:00:00Z",
            }
        write_json(self.spec, spec)
        report = build_package(self.spec, self.package)
        self.assertEqual(report["status"], "pass")

    def test_build_can_select_exact_groups(self) -> None:
        report = build_package(
            self.spec,
            self.package,
            group_ids=["group-002"],
        )
        self.assertEqual(report["group_count"], 1)
        self.assertEqual(report["sample_count"], 3)
        contract = read_json(self.package / "dataset.json")
        self.assertEqual(contract["selection"]["method"], "explicit")
        self.assertEqual(
            contract["selection"]["semantic_groups"],
            ["group-002"],
        )

    def test_nested_groups_are_the_single_spec_source(self) -> None:
        spec = read_json(self.spec)
        grouped: dict[str, dict[str, object]] = {}
        for sample in spec.pop("samples"):
            group_id = sample.pop("semantic_group_id")
            content_category = sample.pop("content_category")
            split = sample.pop("split")
            prompt = sample.pop("prompt")
            group = grouped.get(group_id)
            if group is None:
                group = {
                    "semantic_group_id": group_id,
                    "content_category": content_category,
                    "split": split,
                    "prompt": prompt,
                    "samples": [],
                }
                grouped[group_id] = group
            group["samples"].append(sample)
        spec["groups"] = list(grouped.values())
        self.spec.write_text(json.dumps(spec), encoding="utf-8")
        self.assertEqual(
            [group["semantic_group_id"] for group in list_spec_groups(self.spec)],
            ["group-001", "group-002"],
        )
        report = build_package(
            self.spec,
            self.package,
            group_ids=["group-001"],
        )
        self.assertEqual(report["sample_count"], 3)

    def test_full_build_enforces_real_run_target(self) -> None:
        spec = read_json(self.spec)
        spec["dataset"]["target_group_count"] = 3
        self.spec.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaisesRegex(PipelineError, "below dataset.target_group_count"):
            build_package(self.spec, self.package)
        report = build_package(
            self.spec,
            self.package,
            group_ids=["group-001"],
        )
        self.assertEqual(report["group_count"], 1)

    def test_incomplete_group_is_rejected_before_writing(self) -> None:
        spec = read_json(self.spec)
        spec["samples"].pop()
        self.spec.write_text(json.dumps(spec), encoding="utf-8")
        with self.assertRaisesRegex(PipelineError, "incomplete slot matrix"):
            build_package(self.spec, self.package)
        self.assertFalse(self.package.exists())

    def test_tampered_original_fails_checksum(self) -> None:
        build_package(self.spec, self.package)
        original = next((self.package / "originals").rglob("*.png"))
        original.write_bytes(original.read_bytes() + b"tampered")
        status, report = validate_package(self.package)
        self.assertEqual(status, 1)
        self.assertTrue(
            any("checksum mismatch" in error for error in report["errors"]),
            report,
        )

    def test_tampered_parquet_fails_checksum(self) -> None:
        build_package(self.spec, self.package)
        parquet = next((self.package / "data").glob("*.parquet"))
        parquet.write_bytes(parquet.read_bytes() + b"tampered")
        status, report = validate_package(self.package)
        self.assertEqual(status, 1)
        self.assertTrue(
            any(
                "checksum mismatch" in error or "cannot read Parquet" in error
                for error in report["errors"]
            ),
            report,
        )

    def test_validator_remains_compatible_with_schema_2_jsonl(self) -> None:
        build_package(self.spec, self.package)
        parquet = next((self.package / "data").glob("*.parquet"))
        records = read_parquet_records(parquet)
        for record in records:
            relative = (
                Path("data")
                / record["split"]
                / "images"
                / record["normalized_file_name"]
            )
            target = self.package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(record.pop("image")["bytes"])
            record["schema_version"] = "2.0.0"
            record["normalized_path"] = relative.as_posix()
            record.pop("normalized_file_name")
            record.pop("data_file")
        metadata = self.package / "data/train/metadata.jsonl"
        write_jsonl(metadata, records)
        contract = read_json(self.package / "dataset.json")
        contract["schema_version"] = "2.0.0"
        contract["files"].pop("data")
        contract["files"]["metadata"] = {
            "train": "data/train/metadata.jsonl",
        }
        write_json(self.package / "dataset.json", contract)
        write_json(
            self.package / "groups.json",
            group_index_from_records(
                contract["dataset_id"],
                records,
                schema_version="2.0.0",
            ),
        )
        status, report = validate_package(self.package)
        self.assertEqual(status, 0, report)

    def test_cli_build_and_validate(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mai.dataset_cli",
                "build",
                "--spec",
                str(self.spec),
                "--output",
                str(self.package),
            ],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout[result.stdout.index("{") :])
        self.assertEqual(report["sample_count"], 6)

    def test_publish_validates_then_uploads_and_tags(self) -> None:
        build_package(self.spec, self.package)

        class Result:
            oid = "abc123"

        class Api:
            def __init__(self) -> None:
                self.calls: list[tuple[str, dict[str, object]]] = []

            def create_repo(self, **kwargs: object) -> None:
                self.calls.append(("create_repo", kwargs))

            def upload_folder(self, **kwargs: object) -> Result:
                self.calls.append(("upload_folder", kwargs))
                return Result()

            def create_tag(self, **kwargs: object) -> None:
                self.calls.append(("create_tag", kwargs))

        api = Api()
        report = publish_package(
            self.package,
            "owner/mai-pilot",
            tag="pilot-v1",
            api=api,
        )
        self.assertEqual(report["commit_sha"], "abc123")
        self.assertEqual(
            [name for name, _ in api.calls],
            ["create_repo", "upload_folder", "create_tag"],
        )
        self.assertEqual(api.calls[-1][1]["revision"], "abc123")

    def test_pull_exact_group_rebuilds_valid_subset(self) -> None:
        build_package(self.spec, self.package)

        def downloader(**kwargs: object) -> str:
            return str(self.package / str(kwargs["filename"]))

        groups = list_remote_groups(
            "owner/mai-pilot",
            "abc123",
            downloader=downloader,
        )
        self.assertEqual([group["semantic_group_id"] for group in groups], [
            "group-001",
            "group-002",
        ])
        output = self.root / "nested" / "selected"
        report = pull_groups(
            "owner/mai-pilot",
            "abc123",
            output,
            ["group-002"],
            downloader=downloader,
        )
        self.assertEqual(report["group_count"], 1)
        self.assertEqual(report["sample_count"], 3)
        contract = read_json(output / "dataset.json")
        self.assertEqual(contract["source"]["revision"], "abc123")
        self.assertEqual(contract["selection"]["semantic_groups"], ["group-002"])


if __name__ == "__main__":
    unittest.main()
