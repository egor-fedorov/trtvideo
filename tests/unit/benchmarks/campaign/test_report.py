from typing import Any

import pytest

from benchmarks.scripts.campaign.report import render_markdown


@pytest.mark.parametrize("external", ("vsgan", "tas"))
@pytest.mark.parametrize("explicit_participants", (True, False))
def test_campaign_markdown_uses_dataset_participants(
    external: str, explicit_participants: bool
) -> None:
    products = {
        "trtvideo": "trtvideo",
        "vstrt": "vs-mlrt",
        external: "VSGAN-tensorrt-docker" if external == "vsgan" else "TheAnimeScripter",
    }
    statistics = {
        f"median_{name}": 1.0
        for name in (
            "fps",
            "wall_time_sec",
            "cpu_cores",
            "cpu_capacity_percent",
            "gpu_utilization_percent",
            "power_w",
            "joules_per_frame",
            "peak_vram_mib",
            "output_bitrate_mbps",
            "output_size_mib",
            "startup_sec",
            "steady_state_frame_loop_sec",
            "finalize_mux_sec",
        )
    }
    summary: dict[str, Any] = {
        "status": "valid",
        "execution_profile": "tuned",
        "publication": {"ready": True, "warnings": [], "errors": []},
        "parameters": {"rounds": 3},
        "implementations": {
            name: {
                "product": product,
                "relative_to_trtvideo_percent": 0,
                "statistics": {**statistics, "values_fps": [1, 1, 1]},
                "stability": {
                    "status": "stable",
                    "full_relative_spread": 0,
                    "consensus": None,
                    "outlier": None,
                },
            }
            for name, product in products.items()
        },
    }
    order = [external, "trtvideo", "vstrt"] if explicit_participants else list(products)
    if explicit_participants:
        summary["participants"] = order

    markdown = render_markdown(summary)

    # Throughput, lifecycle, and stability must all retain original identities and order.
    result_rows = [
        line.split(" | ")[0][2:]
        for line in markdown.splitlines()
        if any(line.startswith(f"| {product} |") for product in products.values())
    ]
    assert result_rows == [products[name] for name in order] * 3
    assert ("TheAnimeScripter" in markdown) is (external == "tas")
    assert ("VSGAN-tensorrt-docker" in markdown) is (external == "vsgan")
