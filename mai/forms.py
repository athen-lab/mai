"""Validated command forms used by the MAI workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shlex
import sys


def resolve(repository: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repository / path


def command_preview(form: object, repository: Path) -> str:
    errors = form.errors(repository)  # type: ignore[attr-defined]
    if errors:
        return "Resolve validation issues to preview the command."
    return shlex.join(form.argv(repository))  # type: ignore[attr-defined]


def base_command() -> list[str]:
    return [sys.executable, "-u", "-m", "mai.dataset_cli"]


@dataclass
class InitForm:
    spec: str = "specs/new.json"
    force: bool = False

    def errors(self, repository: Path) -> list[str]:
        errors: list[str] = []
        if not self.spec.strip():
            errors.append("Spec path is required.")
        path = resolve(repository, self.spec)
        if path.exists() and not self.force:
            errors.append("Spec already exists; enable Replace to overwrite it.")
        return errors

    def argv(self, repository: Path) -> list[str]:
        command = [*base_command(), "init", "--spec", self.spec]
        if self.force:
            command.append("--force")
        return command


@dataclass
class BuildForm:
    spec: str = "specs/v1.json"
    selected_groups: list[str] = field(default_factory=list)
    output: str = ".mai-data/package"
    cache: str = ".mai-data/cache"
    force: bool = False
    dry_run: bool = False

    def errors(self, repository: Path) -> list[str]:
        errors: list[str] = []
        if not self.spec.strip() or not resolve(repository, self.spec).is_file():
            errors.append("Spec must be an existing JSON file.")
        if not self.selected_groups:
            errors.append("Choose at least one semantic group.")
        if not self.output.strip():
            errors.append("Package path is required.")
        elif (
            resolve(repository, self.output).exists()
            and not self.force
            and not self.dry_run
        ):
            errors.append("Package exists; enable Replace or choose another path.")
        if not self.cache.strip():
            errors.append("Automatic cache path is required.")
        return errors

    def argv(self, repository: Path) -> list[str]:
        command = [
            *base_command(),
            "build",
            "--spec",
            self.spec,
            "--output",
            self.output,
            "--cache",
            self.cache,
        ]
        for group_id in self.selected_groups:
            command.extend(["--group-id", group_id])
        if self.force:
            command.append("--force")
        if self.dry_run:
            command.append("--dry-run")
        return command


@dataclass
class ValidateForm:
    package: str = ".mai-data/package"

    def errors(self, repository: Path) -> list[str]:
        if not (resolve(repository, self.package) / "dataset.json").is_file():
            return ["Package must contain dataset.json."]
        return []

    def argv(self, repository: Path) -> list[str]:
        return [*base_command(), "validate", "--package", self.package]


@dataclass
class PublishForm:
    package: str = ".mai-data/package"
    repo_id: str = ""
    revision: str = "main"
    tag: str = ""
    private: bool = True

    def errors(self, repository: Path) -> list[str]:
        errors: list[str] = []
        if not (resolve(repository, self.package) / "dataset.json").is_file():
            errors.append("Package must contain dataset.json.")
        if self.repo_id.count("/") != 1 or any(
            not part for part in self.repo_id.split("/")
        ):
            errors.append("Hub repository must use OWNER/NAME.")
        if not self.revision.strip():
            errors.append("Revision is required.")
        return errors

    def argv(self, repository: Path) -> list[str]:
        command = [
            *base_command(),
            "publish",
            "--package",
            self.package,
            "--repo-id",
            self.repo_id,
            "--revision",
            self.revision,
        ]
        if self.tag.strip():
            command.extend(["--tag", self.tag])
        if self.private:
            command.append("--private")
        return command


@dataclass
class PullForm:
    repo_id: str = ""
    revision: str = ""
    output: str = ".mai-data/download"
    selected_groups: list[str] = field(default_factory=list)
    force: bool = False

    def errors(self, repository: Path) -> list[str]:
        errors: list[str] = []
        if self.repo_id.count("/") != 1 or any(
            not part for part in self.repo_id.split("/")
        ):
            errors.append("Hub repository must use OWNER/NAME.")
        if not self.revision.strip():
            errors.append("Pin a commit SHA or release tag.")
        if not self.output.strip():
            errors.append("Output path is required.")
        elif resolve(repository, self.output).exists() and not self.force:
            errors.append("Output exists; enable Replace or choose another path.")
        if not self.selected_groups:
            errors.append("Choose at least one semantic group.")
        return errors

    def argv(self, repository: Path) -> list[str]:
        command = [
            *base_command(),
            "pull",
            "--repo-id",
            self.repo_id,
            "--revision",
            self.revision,
            "--output",
            self.output,
        ]
        for group_id in self.selected_groups:
            command.extend(["--group-id", group_id])
        if self.force:
            command.append("--force")
        return command
