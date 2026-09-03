from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from trtvideo.compatibility.workflow import (
    CompatibilityOptions,
    CompatibilityWorkflowError,
    OnnxContract,
    WorkflowState,
    WorkflowStep,
    _clean_outputs,
    build_plan,
    inspect_onnx_contract,
    run_streaming_command,
    run_workflow,
    workflow_context,
)
from trtvideo.diagnostics.doctor import CheckResult, DoctorReport


def _source(tmp_path: Path, suffix: str) -> Path:
    source = tmp_path / f"model{suffix}"
    source.write_bytes(b"model")
    return source


def _options(tmp_path: Path, *, source_format: str = "checkpoint") -> CompatibilityOptions:
    suffix = ".pth" if source_format == "checkpoint" else ".onnx"
    return CompatibilityOptions(
        source_format=source_format,  # type: ignore[arg-type]
        source_artifact=_source(tmp_path, suffix),
        model_name="2xExample",
        model_source="https://example.test/model",
        model_license="MIT",
        output_dir=tmp_path / "report",
    )


def _static_contract() -> OnnxContract:
    return OnnxContract(
        dynamic=False,
        input_name="input",
        input_shape=(1, 3, 480, 640),
        output_shape=(1, 3, 960, 1280),
        scale=2,
    )


def _dynamic_contract(scale: int | None = None) -> OnnxContract:
    return OnnxContract(
        dynamic=True,
        input_name="input",
        input_shape=(1, 3, None, None),
        output_shape=(1, 3, None, None),
        scale=scale,
    )


def _doctor() -> DoctorReport:
    return DoctorReport(
        checks=(
            CheckResult("GPU", True, "device 0: Test GPU"),
            CheckResult("Driver", True, "600.1"),
            CheckResult("CUDA", True, "runtime 13.0, driver API 13.0"),
            CheckResult("TensorRT", True, "11.0"),
            CheckResult("NVDEC", True, "libnvcuvid.so.1 available"),
            CheckResult("NVENC", True, "libnvidia-encode.so.1 available"),
            CheckResult("VRAM", True, "20.00 GiB free of 24.00 GiB"),
            CheckResult("Disk", True, "10 GiB free"),
        )
    )


def test_checkpoint_plan_uses_default_fixture_and_complete_pipeline(tmp_path: Path) -> None:
    plan = build_plan(_options(tmp_path))

    assert (plan.width, plan.height, plan.scale) == (1280, 720, None)
    assert [step.key for step in plan.steps] == [
        "doctor",
        "input",
        "export",
        "prepare",
        "engine",
        "smoke",
        "report",
    ]
    assert [step.duration_hint for step in plan.steps] == [
        None,
        None,
        None,
        None,
        "model/GPU-dependent, typically 5-15 minutes",
        None,
        None,
    ]
    input_command = plan.steps[1].command
    assert "--source-cache" in input_command
    assert "--input" not in input_command
    assert "--size" in input_command
    assert input_command[input_command.index("--size") + 1] == "1280x720"
    assert "--scale" not in plan.steps[3].command
    assert plan.paths.source_video not in plan.steps[1].outputs
    report_command = plan.steps[-1].command
    assert report_command[report_command.index("--input-manifest") + 1] == str(
        plan.paths.input_manifest
    )


def test_resume_fingerprint_excludes_volatile_free_vram(tmp_path: Path) -> None:
    plan = build_plan(_options(tmp_path))
    first = workflow_context(plan, _doctor())
    changed_free = DoctorReport(
        checks=tuple(
            CheckResult(check.component, check.passed, "5.00 GiB free of 24.00 GiB")
            if check.component == "VRAM"
            else check
            for check in _doctor().checks
        )
    )

    second = workflow_context(plan, changed_free)

    assert first == second
    assert first["gpu"]["VRAM"] == "24.00 GiB"


def test_checkpoint_rejects_manual_scale_before_execution(tmp_path: Path) -> None:
    base = _options(tmp_path)
    options = CompatibilityOptions(**{**base.__dict__, "scale": 2})

    with pytest.raises(CompatibilityWorkflowError, match="scale is inferred"):
        build_plan(options)


def test_plan_rejects_non_public_model_source_before_execution(tmp_path: Path) -> None:
    base = _options(tmp_path)
    options = CompatibilityOptions(**{**base.__dict__, "model_source": "file:///private/model"})

    with pytest.raises(CompatibilityWorkflowError, match=r"public HTTP\(S\) URL"):
        build_plan(options)


def _fake_onnx_model(
    input_shape: tuple[int | None, ...],
    output_shape: tuple[int | None, ...],
    *,
    scale: str | None = None,
    input_type: int = 1,
    output_type: int = 1,
) -> SimpleNamespace:
    def tensor(name: str, shape: tuple[int | None, ...], element_type: int) -> SimpleNamespace:
        dimensions = [
            SimpleNamespace(dim_value=value or 0, dim_param="dynamic" if value is None else "")
            for value in shape
        ]
        return SimpleNamespace(
            name=name,
            type=SimpleNamespace(
                tensor_type=SimpleNamespace(
                    elem_type=element_type,
                    shape=SimpleNamespace(dim=dimensions),
                )
            ),
        )

    metadata = [] if scale is None else [SimpleNamespace(key="trtvideo.upscale_scale", value=scale)]
    return SimpleNamespace(
        graph=SimpleNamespace(
            input=[tensor("input", input_shape, input_type)],
            output=[tensor("output", output_shape, output_type)],
        ),
        metadata_props=metadata,
    )


def test_inspect_onnx_contract_infers_static_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, ".onnx")
    model = _fake_onnx_model((1, 3, 720, 1280), (1, 3, 1440, 2560))
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(load=lambda path, load_external_data: model),
    )

    contract = inspect_onnx_contract(source)

    assert contract.input_name == "input"
    assert contract.dynamic is False
    assert contract.scale == 2


def test_inspect_onnx_contract_reads_dynamic_scale_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, ".onnx")
    model = _fake_onnx_model(
        (1, 3, None, None),
        (1, 3, None, None),
        scale="4",
    )
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(load=lambda path, load_external_data: model),
    )

    contract = inspect_onnx_contract(source)

    assert contract.dynamic is True
    assert contract.scale == 4


def test_inspect_onnx_contract_rejects_conflicting_scale_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, ".onnx")
    model = _fake_onnx_model((1, 3, 720, 1280), (1, 3, 1440, 2560), scale="4")
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(load=lambda path, load_external_data: model),
    )

    with pytest.raises(CompatibilityWorkflowError, match="disagree"):
        inspect_onnx_contract(source)


def test_inspect_onnx_contract_rejects_non_float_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, ".onnx")
    model = _fake_onnx_model(
        (1, 3, 720, 1280),
        (1, 3, 1440, 2560),
        input_type=2,
    )
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(load=lambda path, load_external_data: model),
    )

    with pytest.raises(CompatibilityWorkflowError, match="FP32 or FP16"):
        inspect_onnx_contract(source)


def test_inspect_onnx_contract_wraps_parser_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, ".onnx")

    def fail_load(_path, *, load_external_data):
        assert load_external_data is False
        raise RuntimeError("protobuf decode failure")

    monkeypatch.setitem(sys.modules, "onnx", SimpleNamespace(load=fail_load))

    with pytest.raises(CompatibilityWorkflowError, match="Cannot load ONNX"):
        inspect_onnx_contract(source)


def test_static_onnx_uses_graph_resolution_and_skips_preparation(tmp_path: Path) -> None:
    options = _options(tmp_path, source_format="onnx")
    plan = build_plan(options, onnx_inspector=lambda _path: _static_contract())

    assert (plan.width, plan.height, plan.scale) == (640, 480, 2)
    assert [step.key for step in plan.steps] == ["doctor", "input", "engine", "smoke", "report"]
    engine_command = plan.steps[2].command
    assert str(options.source_artifact) in engine_command


def test_engine_build_uses_selected_gpu(tmp_path: Path) -> None:
    base = _options(tmp_path, source_format="onnx")
    options = CompatibilityOptions(**{**base.__dict__, "gpu_id": 3})

    plan = build_plan(options, onnx_inspector=lambda _path: _static_contract())

    engine_command = plan.steps[2].command
    assert engine_command[:3] == ("env", "CUDA_VISIBLE_DEVICES=3", "build-engine")


def test_dynamic_onnx_requires_scale_when_metadata_cannot_prove_it(tmp_path: Path) -> None:
    options = _options(tmp_path, source_format="onnx")

    with pytest.raises(CompatibilityWorkflowError, match="pass --scale"):
        build_plan(options, onnx_inspector=lambda _path: _dynamic_contract())


def test_dynamic_onnx_uses_explicit_scale_and_preparation(tmp_path: Path) -> None:
    base = _options(tmp_path, source_format="onnx")
    options = CompatibilityOptions(**{**base.__dict__, "scale": 4, "verbose": True})

    plan = build_plan(options, onnx_inspector=lambda _path: _dynamic_contract())

    assert [step.key for step in plan.steps] == [
        "doctor",
        "input",
        "prepare",
        "engine",
        "smoke",
        "report",
    ]
    prepare = plan.steps[2].command
    assert prepare[prepare.index("--scale") + 1] == "4"
    assert "--verbose" in prepare
    assert "--verbose" in plan.steps[3].command


def test_custom_input_controls_checkpoint_resolution(tmp_path: Path) -> None:
    base = _options(tmp_path)
    custom = tmp_path / "custom.mp4"
    custom.write_bytes(b"video")
    options = CompatibilityOptions(**{**base.__dict__, "input_video": custom})

    plan = build_plan(options, video_probe=lambda path: (1920, 1080))

    assert (plan.width, plan.height) == (1920, 1080)
    assert plan.steps[1].command[plan.steps[1].command.index("--input") + 1] == str(custom)


def test_static_onnx_rejects_mismatched_custom_input(tmp_path: Path) -> None:
    base = _options(tmp_path, source_format="onnx")
    custom = tmp_path / "custom.mp4"
    custom.write_bytes(b"video")
    options = CompatibilityOptions(**{**base.__dict__, "input_video": custom})

    with pytest.raises(CompatibilityWorkflowError, match="does not match"):
        build_plan(
            options,
            onnx_inspector=lambda _path: _static_contract(),
            video_probe=lambda _path: (1280, 720),
        )


def test_checkpoint_rejects_odd_custom_input_dimensions(tmp_path: Path) -> None:
    base = _options(tmp_path)
    custom = tmp_path / "custom.mp4"
    custom.write_bytes(b"video")
    options = CompatibilityOptions(**{**base.__dict__, "input_video": custom})

    with pytest.raises(CompatibilityWorkflowError, match="must be even"):
        build_plan(options, video_probe=lambda _path: (1279, 720))


def test_dry_run_has_no_filesystem_or_gpu_side_effects(tmp_path: Path) -> None:
    base = _options(tmp_path)
    options = CompatibilityOptions(**{**base.__dict__, "dry_run": True})
    plan = build_plan(options)

    run_workflow(
        plan,
        doctor_runner=lambda **_kwargs: pytest.fail("doctor must not run"),
        executor=lambda _command: pytest.fail("commands must not run"),
    )

    assert not options.output_dir.exists()


def test_streaming_executor_emits_deterministic_heartbeat() -> None:
    class Process:
        calls = 0

        def wait(self, *, timeout: float) -> int:
            assert timeout == 30
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(("slow",), timeout)
            return 0

    times = iter((10.0, 41.0))
    lines: list[str] = []

    run_streaming_command(
        ("slow",),
        heartbeat_interval=30,
        clock=lambda: next(times),
        popen_factory=lambda _command: Process(),
        emit=lines.append,
    )

    assert lines == ["  still running; elapsed 31s"]


def _artifact_executor(plan, executed: list[str]):
    by_command = {step.command: step for step in plan.steps[1:]}

    def execute(command) -> None:
        step = by_command[tuple(command)]
        executed.append(step.key)
        for output in step.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(f"{step.key}:{output.name}".encode())

    return execute


def test_resume_skips_valid_steps_and_reruns_changed_artifact_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTVIDEO_IMAGE_REF", "trtvideo:model-tools")
    base = _options(tmp_path)
    plan = build_plan(base)
    first: list[str] = []
    run_workflow(
        plan, doctor_runner=lambda **_kwargs: _doctor(), executor=_artifact_executor(plan, first)
    )
    assert first == ["input", "export", "prepare", "engine", "smoke", "report"]

    resumed_options = CompatibilityOptions(**{**base.__dict__, "resume": True})
    resumed_plan = build_plan(resumed_options)
    second: list[str] = []
    run_workflow(
        resumed_plan,
        doctor_runner=lambda **_kwargs: _doctor(),
        executor=_artifact_executor(resumed_plan, second),
    )
    assert second == []

    resumed_plan.paths.input_video.write_bytes(b"changed")
    third: list[str] = []
    run_workflow(
        resumed_plan,
        doctor_runner=lambda **_kwargs: _doctor(),
        executor=_artifact_executor(resumed_plan, third),
    )
    assert third == ["input", "export", "prepare", "engine", "smoke", "report"]


def test_failed_step_is_not_recorded_and_partial_outputs_are_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTVIDEO_IMAGE_REF", "trtvideo:model-tools")
    plan = build_plan(_options(tmp_path))
    input_step = plan.steps[1]

    def fail(command) -> None:
        assert tuple(command) == input_step.command
        for output in input_step.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"partial")
        raise CompatibilityWorkflowError("interrupted")

    with pytest.raises(CompatibilityWorkflowError, match="interrupted"):
        run_workflow(plan, doctor_runner=lambda **_kwargs: _doctor(), executor=fail)

    state = WorkflowState.open(
        plan.paths.state,
        context=workflow_context(plan, _doctor()),
        resume=True,
    )
    assert state.completed_keys == ()
    assert all(not output.exists() for output in input_step.outputs)


def test_invalid_final_report_is_retained_for_diagnosis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTVIDEO_IMAGE_REF", "trtvideo:model-tools")
    plan = build_plan(_options(tmp_path))
    report_step = plan.steps[-1]
    executed: list[str] = []
    successful_executor = _artifact_executor(plan, executed)

    def fail_report(command) -> None:
        if tuple(command) != report_step.command:
            successful_executor(command)
            return
        for output in report_step.outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"invalid diagnostic")
        raise CompatibilityWorkflowError("report invalid")

    with pytest.raises(CompatibilityWorkflowError, match="report invalid"):
        run_workflow(
            plan,
            doctor_runner=lambda **_kwargs: _doctor(),
            executor=fail_report,
        )

    assert all(output.read_bytes() == b"invalid diagnostic" for output in report_step.outputs)
    state = WorkflowState.open(
        plan.paths.state,
        context=workflow_context(plan, _doctor()),
        resume=True,
    )
    assert state.completed_keys == ("input", "export", "prepare", "engine", "smoke")


def test_resume_rejects_changed_context(tmp_path: Path) -> None:
    path = tmp_path / "report" / "compatibility.state.json"
    state = WorkflowState.open(path, context={"gpu": "first"}, resume=False)
    assert state.completed_keys == ()

    with pytest.raises(CompatibilityWorkflowError, match="does not match"):
        WorkflowState.open(path, context={"gpu": "second"}, resume=True)


def test_output_cleanup_refuses_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "report"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    victim = outside / "victim.mp4"
    victim.write_bytes(b"keep")
    (root / "media").symlink_to(outside, target_is_directory=True)
    step = WorkflowStep(
        key="smoke",
        command_name="trtvideo",
        duration_hint="test",
        command=("trtvideo",),
        outputs=(root / "media" / "victim.mp4",),
    )

    with pytest.raises(CompatibilityWorkflowError, match="unsafe path"):
        _clean_outputs(step, root)

    assert victim.read_bytes() == b"keep"
