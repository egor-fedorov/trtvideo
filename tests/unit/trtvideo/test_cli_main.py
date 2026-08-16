from __future__ import annotations

from trtvideo.cli import doctor, process
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
