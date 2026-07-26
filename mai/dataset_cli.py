"""Automation CLI for the MAI Hugging Face dataset pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .dataset import (
    PipelineError,
    build_package,
    initialize_spec,
    validate_package,
    write_json,
)
from .hub import list_remote_groups, publish_package, pull_groups


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="mai-dataset",
        description="Build, validate, publish, and retrieve MAI datasets.",
    )
    commands = root.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create an empty build specification")
    init.add_argument("--spec", type=Path, required=True)
    init.add_argument("--force", action="store_true")

    build = commands.add_parser(
        "build",
        help="build a validated Hugging Face dataset package",
    )
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument(
        "--cache",
        type=Path,
        help="automatic acquisition/generation cache; defaults beside output",
    )
    build.add_argument(
        "--group-id",
        action="append",
        help="build one exact semantic group; repeat to select more",
    )
    build.add_argument("--force", action="store_true")
    build.add_argument("--dry-run", action="store_true")

    validate = commands.add_parser("validate", help="validate a local package")
    validate.add_argument("--package", type=Path, required=True)
    validate.add_argument("--report", type=Path)

    publish = commands.add_parser("publish", help="publish a validated package")
    publish.add_argument("--package", type=Path, required=True)
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--revision", default="main")
    publish.add_argument("--tag")
    publish.add_argument("--private", action="store_true")
    publish.add_argument("--commit-message")

    groups = commands.add_parser("groups", help="list remote semantic groups")
    groups.add_argument("--repo-id", required=True)
    groups.add_argument("--revision", required=True)
    groups.add_argument("--json", action="store_true")

    pull = commands.add_parser("pull", help="retrieve exact groups from the Hub")
    pull.add_argument("--repo-id", required=True)
    pull.add_argument("--revision", required=True)
    pull.add_argument("--output", type=Path, required=True)
    pull.add_argument("--group-id", action="append", required=True)
    pull.add_argument("--force", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            initialize_spec(args.spec, force=args.force)
            result: dict[str, object] = {
                "status": "initialized",
                "spec": str(args.spec.resolve()),
            }
        elif args.command == "build":
            result = build_package(
                args.spec,
                args.output,
                group_ids=args.group_id,
                cache_dir=args.cache,
                force=args.force,
                dry_run=args.dry_run,
            )
        elif args.command == "validate":
            status, result = validate_package(args.package)
            if args.report:
                write_json(args.report, result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return status
        elif args.command == "publish":
            result = publish_package(
                args.package,
                args.repo_id,
                revision=args.revision,
                tag=args.tag,
                private=args.private,
                commit_message=args.commit_message,
            )
        elif args.command == "groups":
            remote_groups = list_remote_groups(args.repo_id, args.revision)
            if args.json:
                print(json.dumps(remote_groups, indent=2))
            else:
                for group in remote_groups:
                    print(
                        f"{group['semantic_group_id']}\t"
                        f"{group.get('content_category', '')}\t"
                        f"{len(group.get('samples', []))} samples"
                    )
            return 0
        elif args.command == "pull":
            result = pull_groups(
                args.repo_id,
                args.revision,
                args.output,
                args.group_id,
                force=args.force,
            )
        else:
            raise AssertionError(args.command)
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
