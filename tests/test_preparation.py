from __future__ import annotations

import binascii
from copy import deepcopy
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zlib

from mai.dataset import PipelineError, build_package
from mai.preparation import (
    _camera_search_queries,
    preparation_plan,
    prepare_groups,
    run_local_diffusers,
)


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + png_chunk(b"IDAT", zlib.compress(row * height))
        + png_chunk(b"IEND", b"")
    )


def fixture_spec() -> tuple[dict[str, object], list[dict[str, object]]]:
    slots = [
        {"slot_id": "camera", "origin_class": "camera"},
        {
            "slot_id": "flux-replicate-0",
            "origin_class": "synthetic",
            "generator_family": "flux",
        },
        {
            "slot_id": "sd-replicate-0",
            "origin_class": "synthetic",
            "generator_family": "stable-diffusion",
        },
    ]
    generators = {
        family: {
            "family_id": family,
            "adapter": "mock-generator",
            "model_id": f"test/{family}",
            "settings": {"width": 1024, "height": 1024},
            "output_terms_url": "https://example.test/terms",
        }
        for family in ("flux", "stable-diffusion")
    }
    groups: list[dict[str, object]] = [
        {
            "semantic_group_id": "group-001",
            "content_category": "object",
            "split": "train",
            "prompt": {
                "prompt_id": "prompt-001",
                "text": "A camera photograph of a red cup on a table.",
                "frozen": True,
            },
        }
    ]
    spec: dict[str, object] = {
        "schema_version": "2.0.0",
        "dataset": {
            "dataset_id": "test-v1",
            "title": "Test",
            "description": "Test preparation",
            "license": "mixed",
            "target_group_count": 1,
            "seed_base": 42,
            "camera_acquisition": {"adapter": "mock-camera"},
            "generators": generators,
            "expected_slots": slots,
        },
        "groups": groups,
    }
    return spec, groups


class PreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_plan_is_network_free_and_reports_local_jobs(self) -> None:
        spec, groups = fixture_spec()
        plan = preparation_plan(spec, groups)
        self.assertEqual(plan["camera_downloads"], 1)
        self.assertEqual(plan["generation_jobs"], 2)
        self.assertEqual(plan["sample_count"], 3)
        self.assertEqual(plan["missing_credentials"], [])

    def test_camera_search_removes_prompt_scaffolding_and_actions(self) -> None:
        group = {
            "prompt": {
                "text": (
                    "A camera photograph of a golden retriever "
                    "standing in green grass."
                )
            }
        }
        self.assertEqual(
            _camera_search_queries(group)[0],
            "golden retriever green grass",
        )

    def test_adapters_prepare_complete_samples_and_reuse_cache(self) -> None:
        spec, groups = fixture_spec()
        calls = {"camera": 0, "generator": 0}
        colors = iter([(200, 20, 20), (20, 200, 20), (20, 20, 200)])

        def camera(
            group: dict[str, object],
            config: dict[str, object],
            target: Path,
        ) -> dict[str, object]:
            calls["camera"] += 1
            make_png(target, next(colors))
            return {
                "source": {
                    "collection_id": "test-camera",
                    "source_record_id": "camera-001",
                    "landing_page_url": "https://example.test/camera-001",
                    "license": {
                        "name": "CC-BY-4.0",
                        "url": "https://creativecommons.org/licenses/by/4.0/",
                    },
                },
                "capture": {
                    "camera_make": "Test",
                    "camera_model": "Camera",
                    "captured_at": "2025-01-01T00:00:00Z",
                    "edit_screen_status": "pass",
                },
                "generation": None,
                "scope": {"in_scope": True, "ambiguity_flags": []},
                "audit": {},
                "provenance": {"kind": "test-camera"},
            }

        def generator(
            group: dict[str, object],
            config: dict[str, object],
            seed: int | None,
            target: Path,
        ) -> dict[str, object]:
            calls["generator"] += 1
            make_png(target, next(colors))
            return {
                "capture": None,
                "generation": {
                    "family_id": config["family_id"],
                    "model_id": config["model_id"],
                    "model_revision": "abc123",
                    "provider": "test",
                    "settings": config["settings"],
                    "input_image_used": False,
                    "seed_status": "recorded",
                    "seed": seed,
                },
                "scope": {"in_scope": True, "ambiguity_flags": []},
                "audit": {},
                "provenance": {"kind": "test-generation", "seed": seed},
            }

        first, _ = prepare_groups(
            spec,
            groups,
            self.root / "cache",
            camera_fetchers={"mock-camera": camera},
            generator_runners={"mock-generator": generator},
        )
        second, _ = prepare_groups(
            spec,
            groups,
            self.root / "cache",
            camera_fetchers={"mock-camera": camera},
            generator_runners={"mock-generator": generator},
        )
        self.assertEqual(len(first), 3)
        self.assertEqual(calls, {"camera": 1, "generator": 2})
        self.assertTrue(all(Path(sample["input_path"]).is_file() for sample in first))
        self.assertTrue(all(sample["audit"]["cache_hit"] for sample in second))
        cached_plan = preparation_plan(spec, groups, self.root / "cache")
        self.assertEqual(cached_plan["cache_hits"], 3)
        self.assertEqual(cached_plan["camera_downloads"], 0)
        self.assertEqual(cached_plan["generation_jobs"], 0)
        seeds = [
            sample["generation"]["seed"]
            for sample in first
            if sample["origin_class"] == "synthetic"
        ]
        self.assertEqual(len(set(seeds)), 2)

    def test_group_driven_build_dry_run_never_calls_adapters(self) -> None:
        spec, _ = fixture_spec()
        spec_path = self.root / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        with patch(
            "mai.preparation.prepare_groups",
            side_effect=AssertionError("must not run"),
        ):
            report = build_package(
                spec_path,
                self.root / "package",
                group_ids=["group-001"],
                dry_run=True,
            )
        self.assertEqual(report["status"], "dry-run")
        self.assertEqual(report["sample_count"], 3)
        self.assertFalse((self.root / "package").exists())

    def test_duplicate_camera_sources_fail_before_generation(self) -> None:
        spec, groups = fixture_spec()
        second = deepcopy(groups[0])
        second["semantic_group_id"] = "group-002"
        second["prompt"]["prompt_id"] = "prompt-002"
        groups.append(second)
        spec["groups"] = groups
        calls = {"camera": 0, "generator": 0}

        def camera(
            group: dict[str, object],
            config: dict[str, object],
            target: Path,
        ) -> dict[str, object]:
            calls["camera"] += 1
            make_png(target, (calls["camera"], 0, 0))
            return {
                "source": {
                    "collection_id": "test-camera",
                    "source_record_id": "same-record",
                    "landing_page_url": "https://example.test/same-record",
                    "license": {
                        "name": "CC-BY-4.0",
                        "url": "https://creativecommons.org/licenses/by/4.0/",
                    },
                },
                "capture": {
                    "camera_make": "Test",
                    "camera_model": "Camera",
                    "captured_at": "2025-01-01T00:00:00Z",
                    "edit_screen_status": "pass",
                },
                "generation": None,
                "scope": {"in_scope": True, "ambiguity_flags": []},
                "audit": {},
                "provenance": {"kind": "test-camera"},
            }

        def generator(
            group: dict[str, object],
            config: dict[str, object],
            seed: int | None,
            target: Path,
        ) -> dict[str, object]:
            calls["generator"] += 1
            raise AssertionError("generation must not start")

        with self.assertRaisesRegex(PipelineError, "camera source duplicates"):
            prepare_groups(
                spec,
                groups,
                self.root / "cache",
                camera_fetchers={"mock-camera": camera},
                generator_runners={"mock-generator": generator},
            )
        self.assertEqual(calls, {"camera": 2, "generator": 0})

    def test_missing_credentials_fail_before_adapter_calls(self) -> None:
        spec, groups = fixture_spec()
        spec["dataset"]["generators"]["flux"]["credential_env"] = "MAI_TEST_TOKEN"
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(PipelineError, "MAI_TEST_TOKEN"):
                prepare_groups(
                    spec,
                    groups,
                    self.root / "cache",
                )

    def test_local_diffusers_adapter_records_revision_device_and_seed(self) -> None:
        generated = self.root / "generated.png"
        observed: dict[str, object] = {}

        class FakeGenerator:
            def __init__(self, device: str) -> None:
                observed["generator_device"] = device

            def manual_seed(self, seed: int) -> FakeGenerator:
                observed["seed"] = seed
                return self

        class FakeImage:
            def convert(self, mode: str) -> FakeImage:
                observed["image_mode"] = mode
                return self

            def save(self, target: Path, format: str) -> None:
                observed["format"] = format
                make_png(target, (10, 20, 30))

        class FakePipeline:
            @classmethod
            def from_pretrained(
                cls,
                model_id: str,
                **kwargs: object,
            ) -> FakePipeline:
                observed["model_id"] = model_id
                observed["load"] = kwargs
                return cls()

            def to(self, device: str) -> None:
                observed["device"] = device

            def set_progress_bar_config(self, **kwargs: object) -> None:
                observed["progress"] = kwargs

            def __call__(
                self,
                prompt: str,
                **kwargs: object,
            ) -> object:
                observed["prompt"] = prompt
                observed["generation"] = kwargs
                return types.SimpleNamespace(images=[FakeImage()])

        class FakeHfApi:
            def model_info(
                self,
                model_id: str,
                revision: str | None = None,
            ) -> object:
                observed["revision_request"] = (model_id, revision)
                return types.SimpleNamespace(sha="repository-sha")

        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.float32 = "float32"
        fake_torch.Generator = FakeGenerator
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        )
        fake_torch.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False),
        )
        fake_diffusers = types.ModuleType("diffusers")
        fake_diffusers.DiffusionPipeline = FakePipeline
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = FakeHfApi
        config = {
            "family_id": "flux",
            "model_id": "test/flux",
            "device": "auto",
            "settings": {
                "width": 512,
                "height": 512,
                "num_inference_steps": 4,
            },
        }
        group = {
            "prompt": {
                "prompt_id": "prompt-001",
                "text": "A camera photograph of a red cup.",
                "frozen": True,
            }
        }
        from mai import preparation

        preparation._LOCAL_PIPELINE.clear()
        preparation._LOCAL_MODEL_REVISIONS.clear()
        with (
            patch.dict(sys.modules, {
                "torch": fake_torch,
                "diffusers": fake_diffusers,
                "huggingface_hub": fake_hub,
            }),
        ):
            fields = run_local_diffusers(group, config, 12345, generated)
        self.assertTrue(generated.is_file())
        self.assertEqual(observed["seed"], 12345)
        self.assertEqual(observed["device"], "cpu")
        self.assertEqual(fields["generation"]["seed"], 12345)
        self.assertEqual(fields["generation"]["provider"], "local-diffusers")
        self.assertEqual(
            fields["generation"]["model_revision"],
            "repository-sha",
        )

    def test_checked_in_v1_smoke_plan_is_three_by_seven(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        report = build_package(
            repository / "specs/v1.json",
            self.root / "package",
            group_ids=[
                "animals-001",
                "architecture-001",
                "food-005",
            ],
            dry_run=True,
        )
        self.assertEqual(report["group_count"], 3)
        self.assertEqual(report["sample_count"], 21)
        self.assertEqual(report["preparation"]["camera_downloads"], 3)
        self.assertEqual(report["preparation"]["generation_jobs"], 18)


if __name__ == "__main__":
    unittest.main()
