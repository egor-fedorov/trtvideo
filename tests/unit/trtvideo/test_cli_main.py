from __future__ import annotations

from types import SimpleNamespace

import pytest

from trtvideo.cli import compatibility_check, compatibility_report, doctor, process
from trtvideo.cli.main import main


def test_dispatcher_routes_doctor_arguments(monkeypatch) -> None:
    received: list[str] = []

    def fake_doctor(arguments) -> int:
        received.extend(arguments)
        return 7

    monkeypatch.setattr(doctor, "main", fake_doctor)

    assert main(["doctor", "--gpu-id", "1"]) == 7
    assert received == ["--gpu-id", "1"]


def test_dispatcher_preserves_processing_arguments(monkeypatch) -> None:
    received: list[str] = []

    def fake_process(arguments) -> None:
        received.extend(arguments)

    monkeypatch.setattr(process, "main", fake_process)

    assert main(["--engine", "model.engine", "--input", "input.mp4"]) is None
    assert received == ["--engine", "model.engine", "--input", "input.mp4"]


def test_dispatcher_routes_compatibility_report_arguments(monkeypatch) -> None:
    received: list[str] = []

    def fake_report(arguments) -> int:
        received.extend(arguments)
        return 2

    monkeypatch.setattr(compatibility_report, "main", fake_report)

    assert main(["compatibility-report", "--model-name", "example"]) == 2
    assert received == ["--model-name", "example"]


def test_dispatcher_routes_compatibility_check_arguments(monkeypatch) -> None:
    received: list[str] = []

    def fake_check(arguments) -> int:
        received.extend(arguments)
        return 3

    monkeypatch.setattr(compatibility_check, "main", fake_check)

    assert main(["compatibility-check", "--checkpoint", "model.pth"]) == 3
    assert received == ["--checkpoint", "model.pth"]


def test_compatibility_check_rejects_checkpoint_in_production_image(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("TRTVIDEO_IMAGE_VARIANT", "production")

    result = compatibility_check.main(
        [
            "--checkpoint",
            str(tmp_path / "model.pth"),
            "--model-name",
            "example",
            "--model-source",
            "https://example.test/model",
            "--model-license",
            "MIT",
            "--output-dir",
            str(tmp_path / "report"),
        ]
    )

    assert result == 2
    assert "Checkpoint compatibility requires the model-tools image" in capsys.readouterr().err


def test_compatibility_check_accepts_static_onnx_in_production_image(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    plan = SimpleNamespace(steps=(SimpleNamespace(key="engine"),))
    executed = []
    monkeypatch.setenv("TRTVIDEO_IMAGE_VARIANT", "production")
    monkeypatch.setattr(compatibility_check, "build_plan", lambda _options: plan)
    monkeypatch.setattr(compatibility_check, "run_workflow", executed.append)

    result = compatibility_check.main(
        [
            "--onnx",
            str(source),
            "--model-name",
            "example",
            "--model-source",
            "https://example.test/model",
            "--model-license",
            "MIT",
            "--output-dir",
            str(tmp_path / "report"),
        ]
    )

    assert result == 0
    assert executed == [plan]


def test_compatibility_check_rejects_dynamic_onnx_in_production_image(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")
    plan = SimpleNamespace(steps=(SimpleNamespace(key="prepare"),))
    monkeypatch.setenv("TRTVIDEO_IMAGE_VARIANT", "production")
    monkeypatch.setattr(compatibility_check, "build_plan", lambda _options: plan)
    monkeypatch.setattr(
        compatibility_check,
        "run_workflow",
        lambda _plan: pytest.fail("dynamic ONNX must not execute"),
    )

    result = compatibility_check.main(
        [
            "--onnx",
            str(source),
            "--model-name",
            "example",
            "--model-source",
            "https://example.test/model",
            "--model-license",
            "MIT",
            "--output-dir",
            str(tmp_path / "report"),
        ]
    )

    assert result == 2
    assert "Dynamic ONNX compatibility requires the model-tools image" in capsys.readouterr().err
