import json
from pathlib import Path

from ai_media.models.registry import (
    EngineRegistryEntry,
    format_registry_entries,
    load_engine_registry,
    select_engine_for_video,
)
from ai_media.video.info import VideoInfo


def _manifest(engine_path: str, *, precision: str = "fp16", io_precision: str = "fp16") -> dict:
    return {
        "engine_path": engine_path,
        "precision": precision,
        "io_precision": io_precision,
        "input": {"shape": [1, 3, 720, 1280]},
        "output": {"shape": [1, 3, 1440, 2560]},
        "tensorrt_version": "10.0",
        "engine_sha256": "engine",
        "model_sha256": "model",
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_engine_registry_from_sidecar_manifest(tmp_path: Path) -> None:
    engine = tmp_path / "model.engine"
    engine.touch()
    sidecar = tmp_path / "model.engine.json"
    _write_json(sidecar, _manifest("model.engine"))

    entries = load_engine_registry(str(sidecar))

    assert len(entries) == 1
    assert entries[0].engine_path == str(engine)
    assert entries[0].input_resolution == (1280, 720)
    assert entries[0].output_resolution == (2560, 1440)
    assert entries[0].precision == "fp16"
    assert entries[0].io_precision == "fp16"


def test_load_engine_registry_from_registry_manifest_with_relative_sidecar(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    engines_dir = model_dir / "engines"
    engine = engines_dir / "720.engine"
    engine.parent.mkdir(parents=True)
    engine.touch()
    _write_json(engines_dir / "720.engine.json", _manifest("720.engine"))
    _write_json(
        model_dir / "manifest.json",
        {"schema_version": 1, "engines": [{"manifest_path": "engines/720.engine.json"}]},
    )

    entries = load_engine_registry(str(model_dir))

    assert len(entries) == 1
    assert entries[0].engine_path == str(engine)


def _entry(
    path: Path,
    *,
    precision: str,
    io_precision: str,
    shape: tuple[int, ...],
) -> EngineRegistryEntry:
    path.touch()
    return EngineRegistryEntry(
        engine_path=str(path),
        manifest_path=None,
        precision=precision,
        io_precision=io_precision,
        input_shape=shape,
        output_shape=(1, 3, shape[2] * 2, shape[3] * 2),
    )


def test_select_engine_for_video_prefers_fp16(tmp_path: Path) -> None:
    video = VideoInfo(width=1280, height=720, fps=60.0, fps_str="60/1", nb_frames=100)
    fp32 = _entry(
        tmp_path / "fp32.engine",
        precision="fp32",
        io_precision="fp32",
        shape=(1, 3, 720, 1280),
    )
    fp16 = _entry(
        tmp_path / "fp16.engine",
        precision="fp16",
        io_precision="fp16",
        shape=(1, 3, 720, 1280),
    )

    selected = select_engine_for_video([fp32, fp16], video)

    assert selected == fp16


def test_select_engine_for_video_honors_precision_filters(tmp_path: Path) -> None:
    video = VideoInfo(width=1280, height=720, fps=60.0, fps_str="60/1", nb_frames=100)
    fp32 = _entry(
        tmp_path / "fp32.engine",
        precision="fp32",
        io_precision="fp32",
        shape=(1, 3, 720, 1280),
    )
    fp16 = _entry(
        tmp_path / "fp16.engine",
        precision="fp16",
        io_precision="fp16",
        shape=(1, 3, 720, 1280),
    )

    selected = select_engine_for_video(
        [fp32, fp16],
        video,
        precision="fp32",
        io_precision="fp32",
    )

    assert selected == fp32


def test_select_engine_for_video_ignores_missing_engine(tmp_path: Path) -> None:
    video = VideoInfo(width=1280, height=720, fps=60.0, fps_str="60/1", nb_frames=100)
    missing = EngineRegistryEntry(
        engine_path=str(tmp_path / "missing.engine"),
        manifest_path=None,
        precision="fp16",
        io_precision="fp16",
        input_shape=(1, 3, 720, 1280),
        output_shape=(1, 3, 1440, 2560),
    )

    assert select_engine_for_video([missing], video) is None


def test_format_registry_entries_reports_missing_engines(tmp_path: Path) -> None:
    entry = EngineRegistryEntry(
        engine_path=str(tmp_path / "missing.engine"),
        manifest_path=None,
        precision="fp16",
        io_precision="fp16",
        input_shape=(1, 3, 720, 1280),
        output_shape=(1, 3, 1440, 2560),
    )

    listing = format_registry_entries([entry])

    assert "1280x720 -> 2560x1440" in listing
    assert "missing" in listing
