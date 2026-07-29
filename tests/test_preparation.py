from __future__ import annotations

import binascii
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
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
    normalize_caption,
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


def make_pattern_png(path: Path, seed: int, size: int = 16) -> None:
    rows = []
    for y in range(size):
        pixels = bytearray()
        for x in range(size):
            pixels.extend(
                (
                    (x * seed + y * 3) % 256,
                    (y * seed + x * 5) % 256,
                    ((x + y) * seed + x * y) % 256,
                )
            )
        rows.append(b"\x00" + bytes(pixels))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0),
        )
        + png_chunk(b"IDAT", zlib.compress(b"".join(rows)))
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

    def test_camera_search_removes_scaffolding_but_keeps_actions(self) -> None:
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
            "golden retriever standing in green grass",
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

    def test_explicit_review_generates_candidates_then_reuses_them(self) -> None:
        spec, groups = fixture_spec()
        spec["dataset"]["prompt_policy"] = {
            "template_id": "natural-photo-test",
            "template": "A natural photograph depicting {concept}.",
        }
        spec["dataset"]["generation_review"] = {
            "candidates_per_slot": 2,
            "require_explicit_decision": True,
            "selection_method": "explicit-first-passing-v1",
        }
        for config in spec["dataset"]["generators"].values():
            config["model_revision"] = "pinned-revision"
        calls = {"camera": 0, "generator": 0}

        def camera(
            group: dict[str, object],
            config: dict[str, object],
            target: Path,
        ) -> dict[str, object]:
            calls["camera"] += 1
            make_png(target, (200, 20, 20))
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
            make_png(target, (20, calls["generator"], 20))
            return {
                "capture": None,
                "generation": {
                    "family_id": config["family_id"],
                    "model_id": config["model_id"],
                    "model_revision": config["model_revision"],
                    "provider": "test",
                    "settings": config["settings"],
                    "input_image_used": False,
                    "seed_status": "recorded",
                    "seed": seed,
                    "rendered_prompt": group["prompt"]["text"],
                    "prompt_template_id": group["prompt_policy"]["template_id"],
                },
                "scope": {
                    "in_scope": False,
                    "ambiguity_flags": ["pending-review"],
                },
                "audit": {"review_status": "pending"},
                "provenance": {
                    "kind": "test-generation",
                    "seed": seed,
                    "prompt": group["prompt"],
                    "concept": group["concept"],
                },
            }

        cache = self.root / "cache"
        with self.assertRaisesRegex(PipelineError, "require review"):
            prepare_groups(
                spec,
                groups,
                cache,
                camera_fetchers={"mock-camera": camera},
                generator_runners={"mock-generator": generator},
            )
        self.assertEqual(calls, {"camera": 1, "generator": 4})
        candidates = json.loads(
            (cache / "review/candidates.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(candidates["candidates"]), 2)
        self.assertTrue((cache / "review/index.html").is_file())
        decisions_path = cache / "review/decisions.json"
        decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
        review_candidates = candidates["candidates"]
        decisions["decisions"]["group-001-flux-replicate-0"] = {
            "status": "accepted",
            "candidate_index": 1,
            "candidate_sha256": review_candidates[
                "group-001-flux-replicate-0"
            ][1]["sha256"],
            "rejected_candidates": {"0": ["semantic_mismatch"]},
            "reviewer": "test-reviewer",
            "reviewed_at": "2026-01-01T00:00:00Z",
        }
        decisions["decisions"]["group-001-sd-replicate-0"] = {
            "status": "accepted",
            "candidate_index": 0,
            "candidate_sha256": review_candidates[
                "group-001-sd-replicate-0"
            ][0]["sha256"],
            "rejected_candidates": {},
            "reviewer": "test-reviewer",
            "reviewed_at": "2026-01-01T00:00:00Z",
        }
        decisions_path.write_text(
            json.dumps(decisions),
            encoding="utf-8",
        )
        samples, plan = prepare_groups(
            spec,
            groups,
            cache,
            camera_fetchers={"mock-camera": camera},
            generator_runners={"mock-generator": generator},
        )
        self.assertEqual(calls, {"camera": 1, "generator": 4})
        self.assertEqual(plan["generation_jobs"], 0)
        flux = next(
            sample
            for sample in samples
            if sample["slot_id"] == "flux-replicate-0"
        )
        self.assertEqual(flux["generation"]["candidate_index"], 1)
        self.assertEqual(flux["audit"]["review_status"], "accepted")
        self.assertEqual(
            flux["prompt"]["text"],
            "A natural photograph depicting a red cup on a table.",
        )

        decisions["decisions"]["group-001-flux-replicate-0"][
            "candidate_sha256"
        ] = "0" * 64
        decisions_path.write_text(
            json.dumps(decisions),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(PipelineError, "candidate_sha256"):
            prepare_groups(
                spec,
                groups,
                cache,
                camera_fetchers={"mock-camera": camera},
                generator_runners={"mock-generator": generator},
            )

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
        self.assertEqual(report["preparation"]["generation_jobs"], 72)
        self.assertEqual(
            report["preparation"]["generation_candidates_per_slot"],
            4,
        )
        self.assertEqual(
            report["preparation"]["review_decisions_pending"],
            18,
        )

    def _v2_fixture(
        self,
    ) -> tuple[
        Path,
        dict[str, object],
        list[dict[str, object]],
        dict[str, object],
    ]:
        repository = Path(__file__).resolve().parents[1]
        spec = json.loads(
            (repository / "specs/v2.json").read_text(encoding="utf-8")
        )
        spec["dataset"]["real_source"].update(
            {
                "image_column": "image",
                "oversample_factor": 2,
                "metadata_filter_columns": [],
                "metadata_reject_patterns": [],
            }
        )
        spec["dataset"]["automated_qa"].update(
            {
                "minimum_dimension": 16,
                "blank_stddev_min": 0.1,
                "near_duplicate_hamming_distance": 0,
                "photo_probability_min": 0.5,
                "unsafe_probability_max": 0.5,
            }
        )
        spec_path = self.root / "v2.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        source_rows: list[dict[str, object]] = []
        source_digests: set[str] = set()
        for index, seed in enumerate((7, 13, 29), 1):
            image = self.root / "source" / f"{index}.png"
            make_pattern_png(image, seed)
            import hashlib

            source_digests.add(hashlib.sha256(image.read_bytes()).hexdigest())
            source_rows.append(
                {
                    "id": f"source-{index}",
                    "image": str(image),
                    "url": f"https://example.test/source-{index}",
                    "license": (
                        "https://creativecommons.org/publicdomain/zero/1.0/"
                    ),
                    "source": "Test archive",
                    "width": 16,
                    "height": 16,
                    "mime_type": "image/png",
                }
            )
        state: dict[str, object] = {
            "source_rows": source_rows,
            "source_digests": source_digests,
            "generator_pass": {},
            "calls": {
                "source": 0,
                "caption": 0,
                "qa": 0,
                "generator": 0,
            },
        }
        return spec_path, spec, spec["groups"], state

    def _v2_adapters(
        self,
        state: dict[str, object],
        *,
        pass_generated: bool = True,
    ) -> tuple[object, object, object, object]:
        import hashlib

        calls = state["calls"]

        def source_loader(
            config: dict[str, object],
            limit: int,
        ) -> list[dict[str, object]]:
            calls["source"] += 1
            return deepcopy(state["source_rows"])[:limit]

        def caption(
            path: Path,
            config: dict[str, object],
        ) -> dict[str, object]:
            calls["caption"] += 1
            return {
                "raw_caption": (
                    "This image shows a beautiful red cup on a wooden table."
                ),
                "runtime": {"mock-captioner": "1"},
            }

        def generator(
            group: dict[str, object],
            config: dict[str, object],
            seed: int | None,
            target: Path,
        ) -> dict[str, object]:
            calls["generator"] += 1
            call_number = calls["generator"]
            make_pattern_png(target, 40 + call_number)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            state["generator_pass"][digest] = (
                pass_generated and call_number % 2 == 0
            )
            return {
                "capture": None,
                "generation": {
                    "family_id": config["family_id"],
                    "model_id": config["model_id"],
                    "model_revision": config["model_revision"],
                    "provider": "mock-generator",
                    "settings": deepcopy(config["settings"]),
                    "input_image_used": False,
                    "seed_status": "recorded",
                    "seed": seed,
                },
                "scope": {
                    "in_scope": False,
                    "ambiguity_flags": ["pending-qa"],
                },
                "audit": {},
                "provenance": {
                    "kind": "mock-generation",
                    "runtime": {"mock-generator": "1"},
                },
            }

        def qa(
            path: Path,
            caption_text: str | None,
            config: dict[str, object],
        ) -> dict[str, object]:
            calls["qa"] += 1
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            source = digest in state["source_digests"]
            passed = source or bool(state["generator_pass"].get(digest))
            return {
                "alignment_score": 0.8 if passed else 0.1,
                "photo_probability": 0.9,
                "unsafe_probability": 0.01,
                "runtime": {"mock-qa": "1"},
            }

        return source_loader, caption, qa, generator

    def test_caption_normalization_is_content_only(self) -> None:
        self.assertEqual(
            normalize_caption(
                "This image shows a stunning red cup, likely on a table."
            ),
            "A red cup, on a table.",
        )

    def test_v2_offline_adapters_are_deterministic_and_first_passing(
        self,
    ) -> None:
        _, spec, groups, state = self._v2_fixture()
        source_loader, caption, qa, generator = self._v2_adapters(state)
        cache = self.root / "cache"
        first, first_plan = prepare_groups(
            spec,
            groups,
            cache,
            source_loaders={"huggingface-dataset": source_loader},
            caption_runners={"moondream-local": caption},
            qa_runners={"clip-local": qa},
            generator_runners={"local-diffusers": generator},
        )
        calls_after_first = deepcopy(state["calls"])
        second, second_plan = prepare_groups(
            spec,
            groups,
            cache,
            source_loaders={"huggingface-dataset": source_loader},
            caption_runners={"moondream-local": caption},
            qa_runners={"clip-local": qa},
            generator_runners={"local-diffusers": generator},
        )
        self.assertEqual(len(first), 21)
        self.assertEqual(len(second), 21)
        self.assertEqual(first_plan["quarantined_group_count"], 0)
        self.assertEqual(second_plan["quarantined_group_count"], 0)
        self.assertEqual(state["calls"]["caption"], calls_after_first["caption"])
        self.assertEqual(
            state["calls"]["generator"],
            calls_after_first["generator"],
        )
        self.assertEqual(state["calls"]["qa"], calls_after_first["qa"])
        synthetic = [
            sample
            for sample in first
            if sample["origin_class"] == "synthetic"
        ]
        self.assertTrue(
            all(
                sample["generation"]["candidate_index"] == 1
                for sample in synthetic
            )
        )
        by_group = {
            sample["semantic_group_id"]: sample
            for sample in first
            if sample["origin_class"] == "real_photo"
        }
        for sample in synthetic:
            source = by_group[sample["semantic_group_id"]]
            self.assertEqual(
                sample["provenance"]["lineage"],
                source["provenance"]["lineage"],
            )
            self.assertEqual(
                sample["provenance"]["captioning"],
                source["provenance"]["captioning"],
            )

    def test_v2_failed_candidates_quarantine_groups(self) -> None:
        _, spec, groups, state = self._v2_fixture()
        source_loader, caption, qa, generator = self._v2_adapters(
            state,
            pass_generated=False,
        )
        samples, plan = prepare_groups(
            spec,
            groups,
            self.root / "quarantine-cache",
            source_loaders={"huggingface-dataset": source_loader},
            caption_runners={"moondream-local": caption},
            qa_runners={"clip-local": qa},
            generator_runners={"local-diffusers": generator},
        )
        self.assertEqual(samples, [])
        self.assertEqual(plan["accepted_group_count"], 0)
        self.assertEqual(plan["quarantined_group_count"], 3)
        for value in plan["quarantine_files"]:
            receipt = json.loads(Path(value).read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "quarantined")
            self.assertEqual(
                receipt["reason"],
                "all_generated_candidates_failed",
            )

    def test_v2_rejects_duplicate_sources_before_split_assignment(self) -> None:
        _, spec, groups, state = self._v2_fixture()
        state["source_rows"][1]["image"] = state["source_rows"][0]["image"]
        source_loader, caption, qa, generator = self._v2_adapters(state)
        with self.assertRaisesRegex(PipelineError, "valid unique photos"):
            prepare_groups(
                spec,
                groups,
                self.root / "duplicate-cache",
                source_loaders={"huggingface-dataset": source_loader},
                caption_runners={"moondream-local": caption},
                qa_runners={"clip-local": qa},
                generator_runners={"local-diffusers": generator},
            )

    def test_checked_in_v2_dry_run_is_network_free(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        report = build_package(
            repository / "specs/v2.json",
            self.root / "v2-package",
            dry_run=True,
        )
        self.assertEqual(report["group_count"], 3)
        self.assertEqual(report["sample_count"], 21)
        self.assertEqual(report["preparation"]["camera_downloads"], 0)
        self.assertEqual(report["preparation"]["caption_jobs"], 3)
        self.assertEqual(report["preparation"]["generation_jobs"], 36)
        self.assertEqual(report["preparation"]["qa_jobs"], 42)
        self.assertFalse((self.root / "v2-package").exists())

    def test_v2_requires_immutable_source_caption_and_qa_revisions(
        self,
    ) -> None:
        _, spec, groups, _ = self._v2_fixture()
        locations = (
            ("real_source", "revision"),
            ("captioning", "model_revision"),
            ("automated_qa", "model_revision"),
        )
        for section, field in locations:
            broken = deepcopy(spec)
            broken["dataset"][section][field] = "main"
            with self.subTest(section=section), self.assertRaisesRegex(
                PipelineError,
                "40-character lowercase commit SHA",
            ):
                preparation_plan(broken, groups)
        broken = deepcopy(spec)
        broken["dataset"]["real_source"]["license_policy"]["allowed"] = []
        with self.assertRaisesRegex(
            PipelineError,
            "license_policy.allowed",
        ):
            preparation_plan(broken, groups)

    @unittest.skipUnless(
        shutil.which("magick")
        and importlib.util.find_spec("datasets") is not None,
        "ImageMagick and Parquet dependencies are required",
    )
    def test_v2_three_group_offline_smoke_package(self) -> None:
        spec_path, _, _, state = self._v2_fixture()
        source_loader, caption, qa, generator = self._v2_adapters(state)
        package = self.root / "v2-package"
        with (
            patch(
                "mai.preparation.load_huggingface_rows",
                source_loader,
            ),
            patch("mai.preparation.run_moondream_caption", caption),
            patch("mai.preparation.run_clip_qa", qa),
            patch("mai.preparation.run_local_diffusers", generator),
        ):
            report = build_package(
                spec_path,
                package,
                cache_dir=self.root / "v2-package-cache",
            )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["group_count"], 3)
        self.assertEqual(report["sample_count"], 21)
        contract = json.loads(
            (package / "dataset.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            contract["design"]["real_source"]["revision"],
            "867988b01138799b89d3ffdd5b4f7e1455951f32",
        )
        self.assertEqual(contract["files"]["quarantine"], [])
        receipts = list((package / "receipts").glob("*.json"))
        self.assertEqual(len(receipts), 21)


if __name__ == "__main__":
    unittest.main()
