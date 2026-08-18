"""Generate deterministic SVG figures from published benchmark JSON."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "rtx-3090"
EXPECTED_FIGURES = (
    "tuned-sweep-light.svg",
    "tuned-sweep-dark.svg",
    "throughput-resources-light.svg",
    "throughput-resources-dark.svg",
)
IMPLEMENTATION_ORDER = ("trtvideo", "vstrt", "vsgan")
IMPLEMENTATION_LABELS = {
    "trtvideo": "trtvideo",
    "vstrt": "vs-mlrt",
    "vsgan": "VSGAN",
}


class FigureDataError(RuntimeError):
    """Published benchmark data cannot support the requested figures."""


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    panel: str
    text: str
    muted_text: str
    grid: str
    project: str
    vstrt: str
    vsgan: str

    def implementation_color(self, implementation: str) -> str:
        return {
            "trtvideo": self.project,
            "vstrt": self.vstrt,
            "vsgan": self.vsgan,
        }[implementation]


THEMES = {
    "light": Theme(
        name="light",
        background="#F8FAFC",
        panel="#F8FAFC",
        text="#14213D",
        muted_text="#526177",
        grid="#CBD5E1",
        project="#0086A8",
        vstrt="#64748B",
        vsgan="#A87952",
    ),
    "dark": Theme(
        name="dark",
        background="#0B1220",
        panel="#0B1220",
        text="#E5EEF8",
        muted_text="#A7B4C5",
        grid="#334155",
        project="#36C5E8",
        vstrt="#94A3B8",
        vsgan="#C9A27E",
    ),
}


@dataclass(frozen=True)
class SweepPoint:
    implementation: str
    candidate_id: str
    streams: int
    fps: float
    winner: bool


@dataclass(frozen=True)
class ResourceLimit:
    implementation: str
    streams: int
    kind: str


@dataclass(frozen=True)
class ImplementationResult:
    implementation: str
    fps: float
    cpu_cores: float
    peak_vram_mib: float


@dataclass(frozen=True)
class WorkloadPanel:
    workload_id: str
    workload: str
    variant: str
    sweep: tuple[SweepPoint, ...]
    resource_limits: tuple[ResourceLimit, ...]
    results: tuple[ImplementationResult, ...]

    @property
    def title(self) -> str:
        return f"{self.workload} | {self.variant} -> {self.output_label}"

    @property
    def output_label(self) -> str:
        return "1440p" if self.variant == "720p" else "4K"

    def result(self, implementation: str) -> ImplementationResult:
        for result in self.results:
            if result.implementation == implementation:
                return result
        raise FigureDataError(f"{self.workload_id}/{self.variant} lacks {implementation} results")

    def fastest_external(self) -> ImplementationResult:
        return max(
            (self.result("vstrt"), self.result("vsgan")),
            key=lambda result: result.fps,
        )


@dataclass(frozen=True)
class PublishedFigureData:
    revision: str
    gpu: str
    cpu: str
    power_limit_w: float
    panels: tuple[WorkloadPanel, ...]

    @property
    def hardware_label(self) -> str:
        return f"{self.gpu} | {self.cpu} | {self.power_limit_w:g} W board limit"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FigureDataError(f"Cannot load published result {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FigureDataError(f"Published result must be a JSON object: {path}")
    return value


def _require_publishable(document: dict[str, Any], path: Path) -> None:
    if document.get("status") != "valid" or document.get("publishable") is not True:
        raise FigureDataError(f"Published result is not valid and publishable: {path}")


def _workload_name(workload_id: str) -> str:
    if workload_id.startswith("realesrgan-"):
        return "RealESRGAN_x2plus"
    if workload_id.startswith("liveaction-span-"):
        return "SPAN"
    raise FigureDataError(f"Unknown published workload: {workload_id}")


def _result_from_json(value: dict[str, Any]) -> ImplementationResult:
    try:
        implementation = str(value["implementation"])
        return ImplementationResult(
            implementation=implementation,
            fps=float(value["fps_median"]),
            cpu_cores=float(value["cpu_cores"]),
            peak_vram_mib=float(value["peak_vram_mib"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FigureDataError(f"Invalid final campaign result: {value}") from exc


def _sweep_from_json(
    selection: dict[str, Any],
) -> tuple[SweepPoint, ...]:
    winners = {
        str(value["candidate_id"])
        for value in selection.get("winners", {}).values()
        if isinstance(value, dict) and "candidate_id" in value
    }
    points: list[SweepPoint] = []
    candidates = selection.get("reconnaissance") or selection.get("candidates", [])
    for candidate in candidates:
        profile = candidate.get("execution_profile", {})
        implementation = str(candidate.get("implementation", ""))
        if implementation not in {"vstrt", "vsgan"}:
            continue
        if candidate.get("status") != "eligible":
            continue
        if profile.get("vapoursynth_threads") != "auto":
            continue
        if profile.get("cuda_graph") is not False:
            continue
        try:
            candidate_id = str(candidate["candidate_id"])
            points.append(
                SweepPoint(
                    implementation=implementation,
                    candidate_id=candidate_id,
                    streams=int(profile["num_streams"]),
                    fps=float(candidate["median_fps"]),
                    winner=candidate_id in winners,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FigureDataError(f"Invalid tuned candidate: {candidate}") from exc
    return tuple(sorted(points, key=lambda point: (point.implementation, point.streams)))


def _resource_limits_from_json(selection: dict[str, Any]) -> tuple[ResourceLimit, ...]:
    limits = selection.get("search", {}).get("resource_limits", {})
    result = []
    for implementation in ("vstrt", "vsgan"):
        value = limits.get(implementation)
        if not isinstance(value, dict):
            continue
        try:
            result.append(
                ResourceLimit(
                    implementation=implementation,
                    streams=int(value["num_streams"]),
                    kind=str(value["kind"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FigureDataError(f"Invalid tuned resource limit: {value}") from exc
    return tuple(result)


def load_published_data(results_dir: Path) -> PublishedFigureData:
    """Load and cross-check the self-contained published result classes."""
    tuned_path = results_dir / "tuned.json"
    tuned = _load_json(tuned_path)
    _require_publishable(tuned, tuned_path)

    tuned_revision = str(tuned.get("scope", {}).get("measurement_revision", ""))
    if not tuned_revision:
        raise FigureDataError("Published tuned results lack a measurement revision")

    try:
        hardware = tuned["environment"]["hardware"]
        gpu = str(hardware["gpu"]["name"])
        cpu = str(hardware["cpu"]["model"])
        power_limit_w = float(hardware["gpu"]["power_limit_w"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FigureDataError("Published tuned results lack hardware identity") from exc

    panels: list[WorkloadPanel] = []
    for workload in tuned.get("workloads", []):
        workload_id = str(workload.get("workload_id", ""))
        variant = str(workload.get("variant", ""))
        final_campaign = workload.get("final_campaign", {})
        if (
            workload.get("status") != "valid"
            or final_campaign.get("status") != "valid"
            or final_campaign.get("publishable") is not True
        ):
            raise FigureDataError(f"Tuned workload is not publishable: {workload_id}/{variant}")
        results = tuple(_result_from_json(result) for result in final_campaign.get("results", []))
        implementations = {result.implementation for result in results}
        if implementations != set(IMPLEMENTATION_ORDER):
            raise FigureDataError(
                f"Unexpected implementations for {workload_id}/{variant}: {sorted(implementations)}"
            )
        panels.append(
            WorkloadPanel(
                workload_id=workload_id,
                workload=_workload_name(workload_id),
                variant=variant,
                sweep=_sweep_from_json(workload.get("selection", {})),
                resource_limits=_resource_limits_from_json(workload.get("selection", {})),
                results=results,
            )
        )

    expected_keys = {
        ("RealESRGAN_x2plus", "720p"),
        ("RealESRGAN_x2plus", "1080p"),
        ("SPAN", "720p"),
        ("SPAN", "1080p"),
    }
    actual_keys = {(panel.workload, panel.variant) for panel in panels}
    if actual_keys != expected_keys:
        raise FigureDataError(f"Published tuned matrix is incomplete: {sorted(actual_keys)}")
    panels.sort(
        key=lambda panel: (
            0 if panel.workload == "RealESRGAN_x2plus" else 1,
            0 if panel.variant == "720p" else 1,
        )
    )
    return PublishedFigureData(
        revision=tuned_revision,
        gpu=gpu,
        cpu=cpu,
        power_limit_w=power_limit_w,
        panels=tuple(panels),
    )


def _configure_matplotlib(theme: Theme) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": theme.muted_text,
            "axes.edgecolor": theme.grid,
            "axes.facecolor": theme.panel,
            "figure.facecolor": theme.background,
            "savefig.facecolor": theme.background,
            "svg.fonttype": "none",
            "svg.hashsalt": "trtvideo-benchmark-figures-v1",
            "text.color": theme.text,
            "xtick.color": theme.muted_text,
            "ytick.color": theme.muted_text,
        }
    )


def _style_panel(ax: Axes, theme: Theme, *, horizontal_grid: bool = True) -> None:
    ax.set_facecolor(theme.panel)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    if horizontal_grid:
        ax.grid(axis="y", color=theme.grid, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def _panel_heading(ax: Axes, panel: WorkloadPanel, theme: Theme) -> None:
    ax.text(
        0.0,
        1.04,
        panel.title,
        transform=ax.transAxes,
        color=theme.text,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _hardware_footer(figure: Figure, data: PublishedFigureData, theme: Theme) -> None:
    figure.text(
        0.99,
        0.015,
        data.hardware_label,
        color=theme.muted_text,
        fontsize=7.5,
        ha="right",
        va="bottom",
    )


def _save_figure(figure: Figure, path: Path, theme: Theme) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        facecolor=theme.background,
        bbox_inches="tight",
        metadata={
            "Creator": "trtvideo benchmark figure generator",
            "Date": None,
        },
    )
    plt.close(figure)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )


def render_tuned_sweep(
    data: PublishedFigureData,
    output_path: Path,
    theme: Theme,
) -> None:
    """Render the complete eligible stream-count sweep as four panels."""
    _configure_matplotlib(theme)
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.2))
    for ax, panel in zip(axes.flat, data.panels, strict=True):
        _style_panel(ax, theme)
        observed_fps: list[float] = []
        for implementation in ("vstrt", "vsgan"):
            points = [point for point in panel.sweep if point.implementation == implementation]
            streams = [point.streams for point in points]
            fps = [point.fps for point in points]
            observed_fps.extend(fps)
            color = theme.implementation_color(implementation)
            ax.plot(
                streams,
                fps,
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=4.5,
            )
            for point in points:
                if point.winner:
                    ax.scatter(
                        [point.streams],
                        [point.fps],
                        s=90,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=1.5,
                        zorder=4,
                    )

        project_fps = panel.result("trtvideo").fps
        observed_fps.append(project_fps)
        ax.axhline(
            project_fps,
            color=theme.project,
            linewidth=1.7,
            linestyle=(0, (5, 3)),
        )

        maximum = max(observed_fps)
        lower = 0.0
        upper = maximum * 1.15
        maximum_streams = max(
            [point.streams for point in panel.sweep]
            + [limit.streams for limit in panel.resource_limits]
        )

        if panel.resource_limits:
            limit_streams = sorted({limit.streams for limit in panel.resource_limits})
            for limit_stream in limit_streams:
                implementations = [
                    IMPLEMENTATION_LABELS[limit.implementation]
                    for limit in panel.resource_limits
                    if limit.streams == limit_stream
                ]
                ax.scatter(
                    [limit_stream],
                    [upper * 0.045],
                    marker="x",
                    s=42,
                    color=theme.text,
                    linewidths=1.4,
                    zorder=5,
                )
                ax.text(
                    limit_stream,
                    upper * 0.075,
                    f"{' / '.join(implementations)} OOM",
                    color=theme.text,
                    fontsize=7.5,
                    ha="center",
                    va="bottom",
                )

        ax.set_xlim(0.8, maximum_streams + 0.2)
        ax.set_ylim(lower, upper)
        ax.set_xticks(range(1, maximum_streams + 1))
        ax.tick_params(axis="both", colors=theme.text)
        ax.set_xlabel("TensorRT streams", color=theme.text)
        ax.set_ylabel("End-to-end FPS", color=theme.text)
        _panel_heading(ax, panel, theme)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=theme.project,
            linewidth=1.7,
            linestyle=(0, (5, 3)),
            label=IMPLEMENTATION_LABELS["trtvideo"],
        ),
        Line2D(
            [0],
            [0],
            color=theme.vstrt,
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label=IMPLEMENTATION_LABELS["vstrt"],
        ),
        Line2D(
            [0],
            [0],
            color=theme.vsgan,
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label=IMPLEMENTATION_LABELS["vsgan"],
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        labelcolor=theme.text,
        bbox_to_anchor=(0.5, 0.995),
    )
    _hardware_footer(figure, data, theme)
    figure.subplots_adjust(left=0.08, right=0.97, top=0.89, bottom=0.12, hspace=0.42)
    _save_figure(figure, output_path, theme)


def render_throughput_resources(
    data: PublishedFigureData,
    output_path: Path,
    theme: Theme,
) -> None:
    """Compare project resource use with the fastest external implementation."""
    _configure_matplotlib(theme)
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    y_positions = list(reversed(range(len(data.panels))))
    labels = [f"{panel.workload}\n{panel.variant} -> {panel.output_label}" for panel in data.panels]
    resource_specs = (
        (
            "Attributed CPU cores",
            lambda result: result.cpu_cores,
            lambda value: f"{value:.2f}",
        ),
        (
            "Peak VRAM (GiB)",
            lambda result: result.peak_vram_mib / 1024.0,
            lambda value: f"{value:.1f}",
        ),
    )
    bar_offset = 0.17
    bar_height = 0.3

    for axis_index, (ax, (title, value_of, format_value)) in enumerate(
        zip(axes, resource_specs, strict=True)
    ):
        _style_panel(ax, theme, horizontal_grid=False)
        ax.grid(axis="x", color=theme.grid, linewidth=0.8, alpha=0.6)
        project_values = [value_of(panel.result("trtvideo")) for panel in data.panels]
        external_results = [panel.fastest_external() for panel in data.panels]
        external_values = [value_of(result) for result in external_results]
        maximum = max(project_values + external_values)

        ax.barh(
            [position + bar_offset for position in y_positions],
            project_values,
            height=bar_height,
            color=theme.project,
            edgecolor="none",
        )
        ax.barh(
            [position - bar_offset for position in y_positions],
            external_values,
            height=bar_height,
            color=theme.vstrt,
            edgecolor="none",
        )

        for index, (position, project_value, external_value) in enumerate(
            zip(y_positions, project_values, external_values, strict=True)
        ):
            ax.text(
                project_value + maximum * 0.018,
                position + bar_offset,
                format_value(project_value),
                color=theme.project,
                fontsize=8,
                va="center",
            )
            external = external_results[index]
            ax.text(
                external_value + maximum * 0.018,
                position - bar_offset,
                f"{format_value(external_value)} {IMPLEMENTATION_LABELS[external.implementation]}",
                color=theme.text,
                fontsize=8,
                va="center",
            )
            if axis_index == 0:
                project_fps = data.panels[index].result("trtvideo").fps
                fps_delta = (project_fps / external.fps - 1.0) * 100.0
                ax.text(
                    max(project_value, external_value) + maximum * 0.16,
                    position,
                    f"{fps_delta:+.1f}% FPS",
                    color=theme.text,
                    fontsize=8.5,
                    fontweight="bold",
                    ha="left",
                    va="center",
                )

        ax.set_xlim(0, maximum * (1.32 if axis_index == 0 else 1.28))
        ax.set_yticks(y_positions, labels)
        ax.set_title(title, color=theme.text, fontsize=11, fontweight="bold", loc="left")
        ax.tick_params(axis="both", colors=theme.text)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in (theme.project, theme.vstrt)]
    figure.legend(
        handles,
        ("trtvideo", "fastest external"),
        loc="lower center",
        ncol=2,
        frameon=False,
        labelcolor=theme.text,
        bbox_to_anchor=(0.5, 0.035),
    )
    figure.text(
        0.25,
        0.09,
        "FPS delta: trtvideo versus the fastest external implementation",
        color=theme.text,
        fontsize=8,
        ha="center",
    )
    _hardware_footer(figure, data, theme)
    figure.subplots_adjust(left=0.18, right=0.98, top=0.9, bottom=0.2, wspace=0.35)
    _save_figure(figure, output_path, theme)


def generate_figures(results_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    """Generate every committed benchmark figure from published JSON."""
    data = load_published_data(results_dir)
    generated: list[Path] = []
    renderers = {
        "tuned-sweep": render_tuned_sweep,
        "throughput-resources": render_throughput_resources,
    }
    for stem, renderer in renderers.items():
        for theme in THEMES.values():
            path = output_dir / f"{stem}-{theme.name}.svg"
            renderer(data, path, theme)
            generated.append(path)
    return tuple(generated)


def check_figures(results_dir: Path, output_dir: Path) -> list[str]:
    """Return missing or stale committed figures without modifying them."""
    with tempfile.TemporaryDirectory(prefix="trtvideo-figures-") as temporary:
        generated_dir = Path(temporary)
        generate_figures(results_dir, generated_dir)
        errors: list[str] = []
        expected_names = set(EXPECTED_FIGURES)
        committed_names = {path.name for path in output_dir.glob("*.svg")}
        for filename in sorted(committed_names - expected_names):
            errors.append(f"unexpected: {output_dir / filename}")
        for filename in EXPECTED_FIGURES:
            expected = output_dir / filename
            actual = generated_dir / filename
            if not expected.exists():
                errors.append(f"missing: {expected}")
            elif expected.read_bytes() != actual.read_bytes():
                errors.append(f"stale: {expected}")
        return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing the published tuned.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="SVG output directory (default: RESULTS_DIR/figures)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when committed SVG files differ; do not modify them",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir is not None else results_dir / "figures"
    )
    if args.check:
        errors = check_figures(results_dir, output_dir)
        if errors:
            for error in errors:
                print(f"ERROR: benchmark figure {error}")
            raise SystemExit(2)
        print(f"Benchmark figures are current: {output_dir}")
        return

    generated = generate_figures(results_dir, output_dir)
    for path in generated:
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
