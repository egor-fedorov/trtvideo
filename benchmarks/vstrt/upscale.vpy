"""Canonical VapourSynth/vstrt full-frame x2 benchmark graph."""

import ctypes
from pathlib import Path

import vapoursynth as vs

core = vs.core
configured_threads = globals().get("vs_threads")
if configured_threads is not None:
    core.num_threads = int(configured_threads)
model_space_stage = globals().get("model_space_stage", "video")
if model_space_stage not in {"video", "input", "output"}:
    raise ValueError(f"Unknown model_space_stage: {model_space_stage}")


def load_raw_rgbs(path, width, height):
    row_bytes = width * 4
    plane_bytes = row_bytes * height
    expected_bytes = plane_bytes * 3
    payload = Path(path).read_bytes()
    if len(payload) != expected_bytes:
        raise ValueError(
            f"Shared RGB tensor has {len(payload)} bytes; expected {expected_bytes}"
        )
    source_buffer = (ctypes.c_ubyte * expected_bytes).from_buffer_copy(payload)
    blank = core.std.BlankClip(
        width=width,
        height=height,
        format=vs.RGBS,
        length=1,
        fpsnum=1,
        fpsden=1,
    )

    def fill_frame(n, f):
        writable = f.copy()
        source_base = ctypes.addressof(source_buffer)
        for plane in range(3):
            destination = ctypes.cast(
                writable.get_write_ptr(plane), ctypes.c_void_p
            ).value
            if destination is None:
                raise ValueError(f"RGBS plane {plane} has a null write pointer")
            stride = writable.get_stride(plane)
            if stride < row_bytes:
                raise ValueError(f"RGBS stride {stride} is smaller than {row_bytes}")
            source_plane = source_base + plane * plane_bytes
            for row in range(height):
                ctypes.memmove(
                    destination + row * stride,
                    source_plane + row * row_bytes,
                    row_bytes,
                )
        return writable

    return core.std.ModifyFrame(clip=blank, clips=blank, selector=fill_frame)


model_input = globals().get("model_input")
if model_input is not None:
    rgb = load_raw_rgbs(
        model_input,
        int(model_width),
        int(model_height),
    )
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
    rgb = core.std.Limiter(rgb, min=0.0, max=1.0)
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
