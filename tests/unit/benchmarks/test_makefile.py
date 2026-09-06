from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def recipe(target: str) -> str:
    source = (ROOT / "benchmarks" / "Makefile").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(target)}:.*\n((?:\t.*\n|\n)+)", source, re.MULTILINE)
    assert match is not None, f"Missing Make recipe: {target}"
    return match.group(1)


def test_tas_image_build_uses_separate_dockerfile() -> None:
    command = recipe("build-tas")
    assert "benchmarks/docker/tas.Dockerfile" in command
    assert "-t $(TAS_IMAGE)" in command
    assert "--build-arg VCS_REF=" in command
    assert "--build-arg VCS_DIRTY=" in command


def test_tas_engine_build_uses_native_builder_cli() -> None:
    command = recipe("build-tas-engine")
    assert "python3 -m benchmarks.scripts.workloads.build_tas_engine" in command
    assert "--onnx /app/$(ONNX)" in command
    assert "--output /app/$(TAS_ENGINE)" in command
    assert "--gpu-id $(GPU_ID)" in command


@pytest.mark.parametrize(
    ("target", "module"),
    [
        ("run-tas", "runners.tas"),
        ("capture-preprocessing-tas", "quality.capture_tas"),
        ("capture-inference-tas", "quality.capture_tas"),
    ],
)
def test_tas_run_and_capture_preserve_io_selection(target: str, module: str) -> None:
    command = recipe(target)
    assert f"python3 -m benchmarks.scripts.{module}" in command
    assert "--network none" in command
    assert "--execution-profile $(EXECUTION_PROFILE)" in command
    assert "$(TAS_ARGS)" in command
    assert "--engine /app/$(TAS_ENGINE)" in command
    assert "--gpu-id $(GPU_ID)" in command
    assert "/models:/app/models:ro" in command
    assert "/videos:/app/videos:ro" in command
    if target == "capture-inference-tas":
        assert "--shared-input-manifest" in command
        assert "/production/trtvideo/manifest.json" in command


@pytest.mark.parametrize("target", ["run-tuned-sweep", "run-tuned-quality", "run-tuned-campaign"])
def test_tuned_stages_receive_tas_engine(target: str) -> None:
    assert '--tas-engine "$(TAS_ENGINE)"' in recipe(target)


def test_tas_preflight_uses_host_python_and_both_engines() -> None:
    command = recipe("preflight-tas-quality")
    assert '"$(HOST_PYTHON)" -m benchmarks.scripts.quality.preflight_tas' in command
    assert '--engine "$(ENGINE)"' in command
    assert '--tas-engine "$(TAS_ENGINE)"' in command
    assert '--root "$(ROOT)"' in command
    assert '--output-dir "$(TUNING_DIR)/tas-preflight"' in command


@pytest.mark.parametrize(
    ("target", "path"),
    [
        ("compare-preprocessing", "/production/tas/manifest.json"),
        ("compare-inference", "/inference/tas/manifest.json"),
        ("compare-product-output", "/tas/run-01/manifest.json"),
    ],
)
def test_quality_comparisons_include_tas(target: str, path: str) -> None:
    assert path in recipe(target)


def test_campaign_receives_tas_io_arguments() -> None:
    assert '--tas-arguments="$(TAS_ARGS)"' in recipe("run-campaign")


def test_tas_retained_outputs_and_rounds_call_runner() -> None:
    assert "$(MAKE) run-tas" in recipe("campaign-tas")
    assert 'TAS_OUTPUT_DIR="$(CAMPAIGN_DIR)/tas/round-$(ROUND)"' in recipe("campaign-tas")
    assert "$(MAKE) run-tas" in recipe("capture-output-tas")
    assert 'TAS_OUTPUT_DIR="$(PRODUCT_OUTPUT_DIR)/tas"' in recipe("capture-output-tas")
