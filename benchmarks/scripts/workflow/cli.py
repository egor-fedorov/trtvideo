"""Run one complete benchmark goal across the canonical matrix."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from benchmarks.scripts.workflow.matrix import (
    WorkflowMatrixError,
    load_workflow_matrix,
)
from benchmarks.scripts.workflow.orchestrator import (
    COMPARATIVE_PROFILE,
    GOALS,
    WorkflowError,
    WorkflowOptions,
    WorkflowState,
    build_plan,
    run_plan,
)

DEFAULT_MATRIX = "benchmarks/workflows/canonical.json"


def _require_supported_python() -> None:
    version = sys.version_info[:2]
    if not (3, 10) <= version < (3, 13):
        raise WorkflowError(
            "Benchmark workflows require host Python >=3.10,<3.13; "
            f"found {version[0]}.{version[1]}. Set HOST_PYTHON to a "
            "supported interpreter."
        )


def _git_output(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise WorkflowError(f"Git command failed: {' '.join(arguments)}") from exc
    return result.stdout.strip()


def _repository_revision(root: Path) -> str:
    revision = _git_output(root, "rev-parse", "HEAD")
    if not revision:
        raise WorkflowError("Cannot determine repository revision")
    return revision


def _require_clean_worktree(root: Path) -> None:
    changes = _git_output(root, "status", "--porcelain")
    if changes:
        raise WorkflowError(
            "Complete benchmark workflows require a clean committed worktree"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_path(
    root: Path,
    *,
    options: WorkflowOptions,
    explicit: str | None,
) -> Path:
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_absolute() else root / path
    scope = (
        f"{options.workload_key or 'all'}-"
        f"{options.variant_name or 'all'}"
    )
    profile = COMPARATIVE_PROFILE if options.goal == "comparative" else "canonical"
    return (
        root
        / "artefacts/benchmarks/workflows"
        / f"{options.goal}-{profile}-{scope}.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("goal", choices=GOALS)
    parser.add_argument(
        "--matrix",
        default=DEFAULT_MATRIX,
        help="Declarative workload/resolution matrix",
    )
    parser.add_argument(
        "--workload",
        default="all",
        help="Workload key from the matrix, or all",
    )
    parser.add_argument(
        "--variant",
        choices=["all", "720p", "1080p"],
        default="all",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the exact workflow state and underlying campaigns",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete ordered plan without checking or executing it",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Override the workflow state path",
    )
    parser.add_argument("--root", default=".")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        _require_supported_python()
        root = Path(args.root).resolve()
        matrix_path = Path(args.matrix)
        if not matrix_path.is_absolute():
            matrix_path = root / matrix_path
        matrix = load_workflow_matrix(matrix_path)
        workload_key = None if args.workload == "all" else args.workload
        if workload_key is not None and workload_key not in matrix.workload_keys:
            raise WorkflowError(
                f"Unknown workload {workload_key!r}; expected one of "
                + ", ".join(matrix.workload_keys)
            )
        options = WorkflowOptions(
            goal=args.goal,
            workload_key=workload_key,
            variant_name=None if args.variant == "all" else args.variant,
            gpu_id=args.gpu_id,
            resume=args.resume,
        )
        plan = build_plan(root=root, matrix=matrix, options=options)
        selections = matrix.select(
            workload_key=options.workload_key,
            variant_name=options.variant_name,
        )
        selection_keys = [selection.key for selection in selections]
        revision = "dry-run" if args.dry_run else _repository_revision(root)
        context = {
            "goal": options.goal,
            "mode": (
                COMPARATIVE_PROFILE
                if options.goal == "comparative"
                else "canonical"
            ),
            "gpu_id": options.gpu_id,
            "matrix_sha256": _sha256(matrix_path),
            "repository_revision": revision,
            "selections": selection_keys,
        }
        state_path = _state_path(
            root,
            options=options,
            explicit=args.state,
        )
        state = None
        if not args.dry_run:
            _require_clean_worktree(root)
            state = WorkflowState.open(
                state_path,
                context=context,
                resume=options.resume,
            )
        print(
            f"Workflow: {options.goal}; "
            f"matrix: {', '.join(selection_keys)}; "
            f"steps: {len(plan)}",
            flush=True,
        )
        if args.dry_run:
            print("Dry run: no commands or state changes will be made", flush=True)
        else:
            print(f"State: {state_path}", flush=True)
        run_plan(
            plan,
            root=root,
            state=state,
            dry_run=args.dry_run,
        )
    except (
        OSError,
        subprocess.CalledProcessError,
        WorkflowError,
        WorkflowMatrixError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    print("Workflow completed successfully", flush=True)


if __name__ == "__main__":
    main()
