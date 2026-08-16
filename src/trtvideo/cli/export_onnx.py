#!/usr/bin/env python3
"""Export a Spandrel-supported x2 image model to static ONNX variants.

Creates ONNX files for selected resolutions, defaulting to 720p and 1080p.

Usage:
    export-onnx --model_path models/pretrained/model.pth --name model
    export-onnx --model_path models/pretrained/model.pth --size 1280x720
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from trtvideo.cli.prepare_onnx import TARGETS, TargetSpec, parse_size
from trtvideo.models.export_conformance import (
    EXPORT_CONTRACT_METADATA_KEY,
    EXPORT_CONTRACT_METADATA_VALUE,
    EXPORT_PROBE_HEIGHT,
    EXPORT_PROBE_WIDTH,
    build_conformance_report,
    compare_outputs,
    conformance_report_path,
    deterministic_probe,
    write_conformance_report,
)

PIXEL_UNSHUFFLE_EXPORT_METADATA_KEY = "trtvideo.pixel_unshuffle_order"
PIXEL_UNSHUFFLE_EXPORT_METADATA_VALUE = "channel-major-v1"
PIXEL_UNSHUFFLE_TRANSPOSE_PERMUTATION = (0, 1, 3, 5, 2, 4)


def load_model(model_path: str) -> Any:
    """Load an image-to-image model supported by Spandrel.

    Args:
        model_path: Path to .pth weights file.

    Returns:
        PyTorch module in eval mode.
    """
    import spandrel

    descriptor = spandrel.ModelLoader().load_from_file(model_path)
    if not isinstance(descriptor, spandrel.ImageModelDescriptor):
        raise TypeError(f"Expected an image-to-image model, got {type(descriptor).__name__}")

    # torch.onnx.export expects torch.nn.Module. Spandrel returns a descriptor wrapper;
    # descriptor.model is the underlying loaded module.
    descriptor.eval()
    model = descriptor.model
    model.eval()
    return model


def freeze_reparameterized_convs(model: Any) -> int:
    """Replace mutable Spandrel Conv3XC blocks with their fused eval convolutions."""
    import torch
    from torch import nn

    replaced = 0

    def freeze_children(module: nn.Module) -> None:
        nonlocal replaced
        for name, child in list(module.named_children()):
            update_params = getattr(child, "update_params", None)
            eval_conv = getattr(child, "eval_conv", None)
            if (
                type(child).__name__ != "Conv3XC"
                or not callable(update_params)
                or not isinstance(eval_conv, nn.Module)
            ):
                freeze_children(child)
                continue

            # Conv3XC normally rewrites fused weights in every eval forward. Capture
            # that equivalent graph once so torch.export sees no module mutations.
            with torch.no_grad():
                update_params()
            layers: list[nn.Module] = [eval_conv]
            if bool(getattr(child, "has_relu", False)):
                layers.append(nn.LeakyReLU(negative_slope=0.05))
            setattr(module, name, nn.Sequential(*layers))
            replaced += 1

    freeze_children(model)
    return replaced


def _translate_pixel_unshuffle(self: Any, downscale_factor: int) -> Any:
    """Translate aten.pixel_unshuffle while preserving PyTorch channel order."""
    from onnxscript import opset18 as op

    input_shape = op.Shape(self)
    batch = op.Slice(input_shape, [0], [1])
    channels = op.Slice(input_shape, [1], [2])
    height = op.Slice(input_shape, [2], [3])
    width = op.Slice(input_shape, [3], [4])
    factor = op.Constant(value_ints=[downscale_factor])
    output_h = op.Div(height, factor)
    output_w = op.Div(width, factor)

    expanded_shape = op.Concat(
        batch,
        channels,
        output_h,
        factor,
        output_w,
        factor,
        axis=0,
    )
    expanded = op.Reshape(self, expanded_shape)
    transposed = op.Transpose(expanded, perm=PIXEL_UNSHUFFLE_TRANSPOSE_PERMUTATION)

    output_channels = op.Mul(channels, op.Mul(factor, factor))
    output_shape = op.Concat(batch, output_channels, output_h, output_w, axis=0)
    return op.Reshape(transposed, output_shape)


def _pixel_unshuffle_translation_table(torch: Any) -> dict[Any, Any]:
    """Return custom ONNX translations required by the export contract."""
    return {
        torch.ops.aten.pixel_unshuffle.default: _translate_pixel_unshuffle,
    }


def _validate_channel_major_unshuffle_graph(model: Any) -> None:
    """Reject the incompatible ONNX lowering that caused RGB channel mixing."""
    if any(node.op_type == "SpaceToDepth" for node in model.graph.node):
        raise RuntimeError(
            "ONNX export lowered PyTorch pixel_unshuffle to SpaceToDepth with "
            "incompatible channel ordering"
        )


def _set_export_metadata(model: Any) -> None:
    import onnx

    properties = {item.key: item.value for item in model.metadata_props}
    properties[EXPORT_CONTRACT_METADATA_KEY] = EXPORT_CONTRACT_METADATA_VALUE
    properties[PIXEL_UNSHUFFLE_EXPORT_METADATA_KEY] = PIXEL_UNSHUFFLE_EXPORT_METADATA_VALUE
    onnx.helper.set_model_props(model, properties)


def export_onnx(
    model: Any,
    input_h: int,
    input_w: int,
    output_path: str,
) -> None:
    """Export model to ONNX with fixed input size.

    Args:
        model: Loaded x2 image model.
        input_h: Input height in pixels.
        input_w: Input width in pixels.
        output_path: Path to save .onnx file.
    """
    import onnx
    import torch

    replaced = freeze_reparameterized_convs(model)
    if replaced:
        print(f"  Reparameterized {replaced} mutable convolution blocks")
    print("  Pixel-unshuffle: channel-major ONNX translation registered")
    dummy_input = torch.randn(1, 3, input_h, input_w, dtype=torch.float32)

    print(f"  Export: input {input_w}x{input_h} -> output {input_w * 2}x{input_h * 2}")
    print(f"  File: {output_path}")

    onnx_program = torch.onnx.export(
        model,
        (dummy_input,),
        f=None,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
        custom_translation_table=_pixel_unshuffle_translation_table(torch),
        # Fixed dimensions — no dynamic_axes
        # This gives TensorRT maximum optimization freedom
    )
    if onnx_program is None:
        raise RuntimeError("PyTorch ONNX exporter did not return an ONNXProgram")

    model_proto = onnx_program.model_proto
    _validate_channel_major_unshuffle_graph(model_proto)
    _set_export_metadata(model_proto)
    onnx.save(model_proto, output_path, convert_attribute=True)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Done: {size_mb:.1f} MB")


def run_export_conformance(model: Any, output_dir: Path) -> dict[str, Any]:
    """Compare the source model with a small ONNX graph from the same exporter."""
    import onnxruntime
    import torch

    torch.set_num_threads(1)
    probe = deterministic_probe(torch)
    with torch.inference_mode():
        reference = model(probe).detach().to(device="cpu", dtype=torch.float32)

    with tempfile.TemporaryDirectory(prefix=".export-probe-", dir=output_dir) as temporary:
        probe_path = Path(temporary) / "probe.onnx"
        export_onnx(
            model,
            EXPORT_PROBE_HEIGHT,
            EXPORT_PROBE_WIDTH,
            str(probe_path),
        )
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session = onnxruntime.InferenceSession(
            str(probe_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(f"Expected one ONNX probe input, got {len(inputs)}")
        outputs = session.run(
            None,
            {inputs[0].name: probe.numpy()},
        )
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one ONNX probe output, got {len(outputs)}")
    candidate = torch.from_numpy(outputs[0])
    return {
        "probe": probe,
        "output_shape": list(reference.shape),
        "metrics": compare_outputs(reference, candidate, torch),
    }


def export_filename(model_name: str, height: int) -> str:
    """Return a deterministic static ONNX filename."""
    return f"{model_name}_{height}p.onnx"


def export_filename_for_target(model_name: str, target: TargetSpec) -> str:
    """Return a deterministic filename for a parsed target resolution."""
    return f"{model_name}_{target['name']}.onnx"


def build_parser() -> argparse.ArgumentParser:
    """Create the exporter CLI parser."""
    parser = argparse.ArgumentParser(
        description="Export a Spandrel-supported x2 image model to static ONNX"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="RealESRGAN_x2plus.pth",
        help="Path to .pth weights file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/onnx",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--name",
        default="realesrgan_x2plus",
        help="Output model basename (default: realesrgan_x2plus)",
    )
    parser.add_argument(
        "--conformance-report",
        type=Path,
        help="Evidence path (default: OUTPUT_DIR/NAME.export-conformance.json)",
    )
    parser.add_argument(
        "--size",
        action="append",
        default=[],
        help="Target input size WIDTHxHEIGHT. Can be repeated. Default: 1280x720 and 1920x1080.",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Verbose output")
    verbosity.add_argument("--quiet", action="store_true", help="Minimal output")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not os.path.exists(args.model_path):
        print(f"ERROR: File {args.model_path} not found.")
        sys.exit(1)

    def log(*a, **kw):
        if not args.quiet:
            print(*a, **kw)

    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)
    report_path = args.conformance_report or conformance_report_path(output_dir, args.name)
    report_path.unlink(missing_ok=True)

    # Load model
    log("Loading model...")
    model = load_model(args.model_path)
    log(f"  Model loaded: {type(model).__name__}")

    log("\n--- Validate source-model export conformance ---")
    conformance = run_export_conformance(
        model,
        output_dir,
    )

    targets = [parse_size(size) for size in args.size] or TARGETS
    exported_paths: list[Path] = []

    for target in targets:
        input_h, input_w = target["h"], target["w"]
        filename = export_filename_for_target(args.name, target)
        output_path = output_dir / filename
        log(f"\n--- Export {filename} ---")
        export_onnx(model, input_h, input_w, str(output_path))
        exported_paths.append(output_path)

    report = build_conformance_report(
        model_name=args.name,
        weights_path=Path(args.model_path),
        probe=conformance["probe"],
        output_shape=conformance["output_shape"],
        metrics=conformance["metrics"],
        exported_paths=exported_paths,
        output_dir=output_dir,
    )
    write_conformance_report(report_path, report)
    log(f"Export conformance: valid ({report_path})")

    log("\n=== All ONNX models exported ===")
    log(f"Files in: {args.output_dir}/")


if __name__ == "__main__":
    main()
