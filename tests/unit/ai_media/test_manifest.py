import pytest

from ai_media.models.manifest import make_upscale_model_spec


def test_make_upscale_model_spec() -> None:
    spec = make_upscale_model_spec(
        name="span",
        input_name="input",
        output_name="output",
        input_shape=(1, 3, 720, 1280),
        output_shape=(1, 3, 1440, 2560),
        input_dtype="fp16",
        output_dtype="fp16",
    )

    assert spec.name == "span"
    assert spec.task == "upscale"
    assert spec.scale == 2
    assert spec.inputs[0].dtype == "fp16"
    assert spec.outputs[0].shape == (1, 3, 1440, 2560)


@pytest.mark.parametrize(
    ("input_shape", "output_shape"),
    [
        ((1, 3, 720), (1, 3, 1440, 2560)),
        ((1, 3, 720, 1280), (1, 3, 1440)),
        ((0, 3, 720, 1280), (1, 3, 1440, 2560)),
        ((2, 3, 720, 1280), (1, 3, 1440, 2560)),
        ((1, 1, 720, 1280), (1, 3, 1440, 2560)),
        ((1, 3, 720, 1280), (1, 3, 1000, 2560)),
        ((1, 3, 720, 1280), (1, 3, 1440, 3840)),
    ],
)
def test_make_upscale_model_spec_rejects_invalid_shapes(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        make_upscale_model_spec(
            name="bad",
            input_name="input",
            output_name="output",
            input_shape=input_shape,
            output_shape=output_shape,
        )
