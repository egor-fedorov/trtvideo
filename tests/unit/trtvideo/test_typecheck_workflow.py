import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _recipe(target: str) -> str:
    source = (ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(target)}:.*\n((?:\t.*\n|\n)+)", source, re.MULTILINE)
    assert match is not None, f"Missing Make recipe: {target}"
    return match.group(1)


def test_runtime_typecheck_build_reuses_full_locked_environment() -> None:
    command = _recipe("build-typecheck-runtime")

    assert "--build-arg INSTALL_DEV=1" in command
    assert "--target benchmark" in command
    assert "-t $(TYPECHECK_RUNTIME_IMAGE)" in command
    assert "--gpus" not in command


def test_runtime_typecheck_checks_dependencies_and_runs_both_checkers() -> None:
    command = _recipe("typecheck-runtime")

    assert "from importlib.metadata import version" in command
    for package in ("tensorrt", "cuda-bindings", "cvcuda-cu12", "pynvvideocodec", "torch"):
        assert package in command
    assert '$(MAKE) -k typecheck DEV_IMAGE="$(TYPECHECK_RUNTIME_IMAGE)"' in command
    assert "--gpus" not in command


def test_full_dependency_checks_stay_outside_lightweight_gate() -> None:
    source = (ROOT / "Makefile").read_text(encoding="utf-8")
    gate = next(line for line in source.splitlines() if line.startswith("check:"))

    assert "typecheck" in gate
    assert "typecheck-runtime" not in gate
    assert "test-export-integration" not in gate
    assert "$(TYPECHECK_RUNTIME_IMAGE)" in _recipe("test-export-integration")
    assert "tests/model_tools" in _recipe("test-export-integration")
    assert "tests/integration" in _recipe("test-media-integration")
