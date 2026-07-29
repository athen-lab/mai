from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mai.dataset import list_spec_groups
from mai.forms import BuildForm, PublishForm, PullForm, command_preview
from mai.tui import OPERATIONS, create_form


class FormTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repository = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_build_is_first_operation(self) -> None:
        self.assertEqual(OPERATIONS[0][0], "build")
        form = create_form("build")
        self.assertIsInstance(form, BuildForm)
        self.assertEqual(form.spec, "specs/v2.json")

    def test_checked_in_v1_supports_the_real_run_matrix(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        spec = json.loads(
            (repository / "specs/v1.json").read_text(encoding="utf-8")
        )
        slots = spec["dataset"]["expected_slots"]
        groups = spec["groups"]
        families = {
            slot["generator_family"]
            for slot in slots
            if slot["origin_class"] == "synthetic"
        }
        adapters = {
            generator["adapter"]
            for generator in spec["dataset"]["generators"].values()
        }
        self.assertEqual(spec["dataset"]["target_group_count"], 200)
        self.assertEqual(len(slots), 7)
        self.assertEqual(len(families), 3)
        self.assertEqual(adapters, {"local-diffusers"})
        self.assertEqual(len(groups), 200)
        self.assertEqual(
            len({group["semantic_group_id"] for group in groups}),
            200,
        )
        self.assertEqual(
            len({group["prompt"]["prompt_id"] for group in groups}),
            200,
        )
        self.assertEqual(
            [sum(group["split"] == split for group in groups) for split in (
                "train",
                "validation",
                "test",
            )],
            [160, 20, 20],
        )

    def test_build_form_preview_uses_automation_cli(self) -> None:
        spec = self.repository / "spec.json"
        spec.write_text("{}", encoding="utf-8")
        form = BuildForm(
            spec="spec.json",
            selected_groups=["group-a"],
            output="package",
            dry_run=True,
        )
        preview = command_preview(form, self.repository)
        self.assertIn("-m mai.dataset_cli build", preview)
        self.assertIn("--group-id group-a", preview)
        self.assertIn("--dry-run", preview)
        self.assertNotIn("--allow-paid-generation", preview)

    def test_v1_selector_labels_on_demand_slots(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        groups = list_spec_groups(repository / "specs/v1.json")
        self.assertEqual(groups[0]["sample_count"], 7)
        self.assertEqual(groups[0]["sample_count_label"], "on demand")

    def test_publish_requires_owner_name(self) -> None:
        package = self.repository / "package"
        package.mkdir()
        (package / "dataset.json").write_text("{}", encoding="utf-8")
        errors = PublishForm(package="package", repo_id="bad").errors(self.repository)
        self.assertIn("Hub repository must use OWNER/NAME.", errors)

    def test_pull_uses_explicit_group_flags(self) -> None:
        form = PullForm(
            repo_id="owner/data",
            revision="abc123",
            output="selected",
            selected_groups=["group-a", "group-c"],
        )
        argv = form.argv(self.repository)
        self.assertEqual(argv.count("--group-id"), 2)
        self.assertNotIn("--groups", argv)

    def test_module_help_does_not_require_a_tty(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "mai", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MAI research workbench", result.stdout)


if __name__ == "__main__":
    unittest.main()
