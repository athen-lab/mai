"""Hugging Face Hub publishing and exact-group retrieval."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .dataset import (
    PipelineError,
    group_index_from_records,
    read_json,
    read_jsonl,
    safe_path,
    validate_package,
    write_json,
    write_jsonl,
)


def _hub_symbols() -> tuple[Any, Any]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise PipelineError(
            "Hugging Face support is not installed. "
            "Install with `python3 -m pip install -e '.[hub]'`."
        ) from error
    return HfApi, hf_hub_download


def publish_package(
    package: Path,
    repo_id: str,
    *,
    revision: str = "main",
    private: bool = False,
    tag: str | None = None,
    token: str | None = None,
    commit_message: str | None = None,
    api: Any | None = None,
) -> dict[str, Any]:
    package = package.resolve()
    status, report = validate_package(package)
    if status:
        raise PipelineError(
            "refusing to publish an invalid package: "
            + "; ".join(report["errors"][:5])
        )
    if not repo_id or "/" not in repo_id:
        raise PipelineError("repo_id must be in OWNER/NAME form")
    if api is None:
        HfApi, _ = _hub_symbols()
        api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    if revision != "main":
        try:
            api.create_branch(
                repo_id=repo_id,
                branch=revision,
                repo_type="dataset",
                exist_ok=True,
            )
        except TypeError:
            api.create_branch(
                repo_id=repo_id,
                branch=revision,
                repo_type="dataset",
            )
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        folder_path=str(package),
        commit_message=commit_message
        or f"Publish validated {report['dataset_id']} dataset",
    )
    commit_sha = getattr(result, "oid", None) or getattr(result, "commit_id", None)
    if tag:
        api.create_tag(
            repo_id=repo_id,
            tag=tag,
            repo_type="dataset",
            revision=commit_sha or revision,
            exist_ok=False,
        )
    return {
        "status": "published",
        "repo_id": repo_id,
        "revision": revision,
        "tag": tag,
        "commit_sha": commit_sha,
        "dataset_id": report["dataset_id"],
        "sample_count": report["sample_count"],
    }


def remote_file(
    repo_id: str,
    filename: str,
    revision: str,
    *,
    token: str | None = None,
    downloader: Any | None = None,
) -> Path:
    if downloader is None:
        _, downloader = _hub_symbols()
    try:
        result = downloader(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
    except Exception as error:
        raise PipelineError(
            f"cannot download {repo_id}@{revision}:{filename}: {error}"
        ) from error
    return Path(result)


def list_remote_groups(
    repo_id: str,
    revision: str,
    *,
    token: str | None = None,
    downloader: Any | None = None,
) -> list[dict[str, Any]]:
    path = remote_file(
        repo_id,
        "groups.json",
        revision,
        token=token,
        downloader=downloader,
    )
    index = read_json(path)
    groups = index.get("groups")
    if not isinstance(groups, list):
        raise PipelineError("remote groups.json has no groups array")
    return groups


def pull_groups(
    repo_id: str,
    revision: str,
    output: Path,
    group_ids: list[str],
    *,
    force: bool = False,
    token: str | None = None,
    downloader: Any | None = None,
) -> dict[str, Any]:
    if not group_ids:
        raise PipelineError("choose at least one semantic group")
    if len(set(group_ids)) != len(group_ids):
        raise PipelineError("semantic group selection contains duplicates")
    output = output.resolve()
    if output.exists():
        if not force:
            raise PipelineError(f"output already exists: {output}; use --force")
        if not (output / "dataset.json").is_file():
            raise PipelineError(f"refusing to replace unrecognized directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    contract_path = remote_file(
        repo_id,
        "dataset.json",
        revision,
        token=token,
        downloader=downloader,
    )
    index_path = remote_file(
        repo_id,
        "groups.json",
        revision,
        token=token,
        downloader=downloader,
    )
    card_path = remote_file(
        repo_id,
        "README.md",
        revision,
        token=token,
        downloader=downloader,
    )
    contract = read_json(contract_path)
    index = read_json(index_path)
    by_group = {
        group["semantic_group_id"]: group for group in index.get("groups", [])
    }
    unknown = sorted(set(group_ids) - set(by_group))
    if unknown:
        raise PipelineError("unknown semantic groups: " + ", ".join(unknown))
    selected_groups = [by_group[group_id] for group_id in group_ids]
    selected_ids = {
        sample["sample_id"]
        for group in selected_groups
        for sample in group["samples"]
    }
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-pull-", dir=output.parent)
    )
    try:
        selected_records: list[dict[str, Any]] = []
        records_by_split: dict[str, list[dict[str, Any]]] = {}
        for split, relative in contract["files"]["metadata"].items():
            remote_metadata = remote_file(
                repo_id,
                relative,
                revision,
                token=token,
                downloader=downloader,
            )
            records = [
                record
                for record in read_jsonl(remote_metadata)
                if record.get("sample_id") in selected_ids
            ]
            if records:
                records_by_split[split] = records
                selected_records.extend(records)
        if {record["sample_id"] for record in selected_records} != selected_ids:
            raise PipelineError("remote metadata is missing selected samples")
        file_paths = {
            relative
            for record in selected_records
            for relative in (
                record["receipt_path"],
                record["original_path"],
                record["normalized_path"],
            )
        }
        for relative in sorted(file_paths):
            cached = remote_file(
                repo_id,
                relative,
                revision,
                token=token,
                downloader=downloader,
            )
            target = safe_path(staging, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
        selected_contract = json.loads(json.dumps(contract))
        selected_contract["source"] = {
            "repo_id": repo_id,
            "revision": revision,
        }
        selected_contract["selection"] = {
            "method": "explicit",
            "semantic_groups": group_ids,
            "sample_count": len(selected_records),
        }
        selected_contract["files"]["metadata"] = {}
        for split, records in records_by_split.items():
            relative = f"data/{split}/metadata.jsonl"
            write_jsonl(staging / relative, records)
            selected_contract["files"]["metadata"][split] = relative
        write_json(staging / "dataset.json", selected_contract)
        write_json(
            staging / "groups.json",
            group_index_from_records(contract["dataset_id"], selected_records),
        )
        shutil.copyfile(card_path, staging / "README.md")
        status, report = validate_package(staging)
        write_json(staging / "validation_report.json", report)
        if status:
            raise PipelineError(
                "pulled package failed validation: "
                + "; ".join(report["errors"][:5])
            )
        if output.exists():
            shutil.rmtree(output)
        os.replace(staging, output)
        return {**report, "output": str(output), "revision": revision}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
