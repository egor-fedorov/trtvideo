"""Canonical VapourSynth/vstrt full-frame x2 benchmark graph."""

import vapoursynth as vs

core = vs.core
configured_threads = globals().get("vs_threads")
if configured_threads is not None:
    core.num_threads = int(configured_threads)
model_space_stage = globals().get("model_space_stage", "video")
if model_space_stage not in {"video", "input", "output"}:
    raise ValueError(f"Unknown model_space_stage: {model_space_stage}")

model_input = globals().get("model_input")
if model_input is not None:
    rgb = core.bs.VideoSource(
        source=model_input,
        cachemode=0,
        showprogress=False,
    )
    if rgb.format is None or rgb.format.id != vs.RGBS:
        raise ValueError("Shared model input must decode as RGBS")
else:
    source_clip = core.bs.VideoSource(
        source=source,
        cachemode=0,
        showprogress=False,
    )
    rgb = core.resize.Bicubic(
        source_clip,
        format=vs.RGBS,
        matrix_in_s="709",
        range_in_s="limited",
    )
if model_space_stage == "input":
    rgb.set_output()
else:
    upscaled = core.trt.Model(
        rgb,
        engine_path=engine,
        device_id=int(gpu_id),
        use_cuda_graph=cuda_graph == "1",
        num_streams=int(num_streams),
        verbosity=2,
    )
    if model_space_stage == "output":
        upscaled.set_output()
    else:
        output = core.resize.Bicubic(
            upscaled,
            format=vs.YUV420P8,
            matrix_s="709",
            range_s="limited",
            dither_type="error_diffusion",
        )
        output = core.std.SetFrameProps(
            output,
            _ColorRange=1,
            _Matrix=1,
            _Transfer=1,
            _Primaries=1,
        )
        output.set_output()
