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

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "rtx-3090"
EXPECTED_FIGURES = (
    "tuned-sweep-light.svg",
    "tuned-sweep-dark.svg",
    "throughput-resources-light.svg",
    "throughput-resources-dark.svg",
    "lifecycle-light.svg",
    "lifecycle-dark.svg",
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
    startup: str
    steady: str
    finalize: str

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
        startup="#D6A84B",
        steady="#0086A8",
        finalize="#94A3B8",
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
        startup="#E9BC62",
        steady="#36C5E8",
        finalize="#64748B",
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
class ImplementationResult:
    implementation: str
    fps: float
    cpu_cores: float
    peak_vram_mib: float
    startup_sec: float
    steady_state_sec: float
    finalize_sec: float

    @property
    def wall_sec(self) -> float:
        return self.startup_sec + self.steady_state_sec + self.finalize_sec


@dataclass(frozen=True)
class WorkloadPanel:
    workload_id: str
    workload: str
    variant: str
    sweep: tuple[SweepPoint, ...]
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


@dataclass(frozen=True)
class PublishedFigureData:
    revision: str
    panels: tuple[WorkloadPanel, ...]


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
            startup_sec=float(value["startup_sec"]),
            steady_state_sec=float(value["steady_state_sec"]),
            finalize_sec=float(value["finalize_sec"]),
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
    for candidate in selection.get("candidates", []):
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


def load_published_data(results_dir: Path) -> PublishedFigureData:
    """Load and cross-check the self-contained published result classes."""
    tuned_path = results_dir / "tuned.json"
    upstream_path = results_dir / "upstream-default.json"
    tuned = _load_json(tuned_path)
    upstream = _load_json(upstream_path)
    _require_publishable(tuned, tuned_path)
    _require_publishable(upstream, upstream_path)

    tuned_revision = str(tuned.get("scope", {}).get("measurement_revision", ""))
    upstream_revision = str(upstream.get("scope", {}).get("measurement_revision", ""))
    if not tuned_revision or tuned_revision != upstream_revision:
        raise FigureDataError("Published result classes use different measurement revisions")

    upstream_keys = {
        (str(campaign["workload_id"]), str(campaign["variant"]))
        for campaign in upstream.get("campaigns", [])
    }
    panels: list[WorkloadPanel] = []
    for workload in tuned.get("workloads", []):
        workload_id = str(workload.get("workload_id", ""))
        variant = str(workload.get("variant", ""))
        if (workload_id, variant) not in upstream_keys:
            raise FigureDataError(f"Upstream-default results lack {workload_id}/{variant}")
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
    return PublishedFigureData(revision=tuned_revision, panels=tuple(panels))


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


def _spread_label_positions(
    values: list[tuple[str, float]],
    lower: float,
    upper: float,
) -> dict[str, float]:
    if not values:
        return {}
    gap = max((upper - lower) * 0.075, 1e-9)
    ordered = sorted(values, key=lambda item: item[1])
    positioned: list[list[Any]] = [[name, value] for name, value in ordered]
    for index in range(1, len(positioned)):
        positioned[index][1] = max(
            positioned[index][1],
            positioned[index - 1][1] + gap,
        )
    overflow = positioned[-1][1] - upper
    if overflow > 0:
        for item in positioned:
            item[1] -= overflow
    underflow = lower - positioned[0][1]
    if underflow > 0:
        for item in positioned:
            item[1] += underflow
    return {str(name): float(value) for name, value in positioned}


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
        endpoints: list[tuple[str, float]] = []
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
            endpoints.append((implementation, fps[-1]))

        project_fps = panel.result("trtvideo").fps
        observed_fps.append(project_fps)
        ax.axhline(
            project_fps,
            color=theme.project,
            linewidth=1.7,
            linestyle=(0, (5, 3)),
        )
        endpoints.append(("trtvideo", project_fps))

        minimum = min(observed_fps)
        maximum = max(observed_fps)
        padding = max((maximum - minimum) * 0.22, maximum * 0.015)
        lower = minimum - padding
        upper = maximum + padding
        labels = _spread_label_positions(endpoints, lower, upper)
        endpoint_lookup = dict(endpoints)
        for implementation in IMPLEMENTATION_ORDER:
            source_x = (
                6
                if implementation == "trtvideo"
                else max(
                    point.streams for point in panel.sweep if point.implementation == implementation
                )
            )
            source_y = endpoint_lookup[implementation]
            ax.annotate(
                IMPLEMENTATION_LABELS[implementation],
                xy=(source_x, source_y),
                xytext=(6.18, labels[implementation]),
                color=theme.implementation_color(implementation),
                fontsize=8.5,
                va="center",
                arrowprops={
                    "arrowstyle": "-",
                    "color": theme.implementation_color(implementation),
                    "linewidth": 0.7,
                },
            )

        ax.set_xlim(1.8, 7.05)
        ax.set_ylim(lower, upper)
        ax.set_xticks((2, 3, 4, 5, 6))
        ax.set_xlabel("TensorRT streams")
        ax.set_ylabel("End-to-end FPS")
        _panel_heading(ax, panel, theme)

    figure.subplots_adjust(left=0.08, right=0.91, top=0.93, bottom=0.09, hspace=0.42)
    _save_figure(figure, output_path, theme)


def render_throughput_resources(
    data: PublishedFigureData,
    output_path: Path,
    theme: Theme,
) -> None:
    """Render FPS against attributed CPU, with bubble area representing VRAM."""
    _configure_matplotlib(theme)
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.2))
    for ax, panel in zip(axes.flat, data.panels, strict=True):
        _style_panel(ax, theme)
        project_fps = panel.result("trtvideo").fps
        ax.axhspan(
            project_fps * 0.95,
            project_fps * 1.05,
            color=theme.project,
            alpha=0.08,
            linewidth=0,
        )
        values = [panel.result(name) for name in IMPLEMENTATION_ORDER]
        for result in values:
            color = theme.implementation_color(result.implementation)
            ax.scatter(
                [result.cpu_cores],
                [result.fps],
                s=result.peak_vram_mib * 0.05,
                color=color,
                alpha=0.88,
                edgecolors=theme.background,
                linewidths=1.0,
                zorder=3,
            )
            offset = {
                "trtvideo": (7, 8),
                "vstrt": (7, 8),
                "vsgan": (7, -13),
            }[result.implementation]
            ax.annotate(
                IMPLEMENTATION_LABELS[result.implementation],
                (result.cpu_cores, result.fps),
                xytext=offset,
                textcoords="offset points",
                color=color,
                fontsize=8.5,
                fontweight="bold" if result.implementation == "trtvideo" else "normal",
            )

        minimum = min(value.fps for value in values)
        maximum = max(value.fps for value in values)
        padding = max((maximum - minimum) * 1.2, project_fps * 0.035)
        ax.set_ylim(minimum - padding, maximum + padding)
        ax.set_xscale("log")
        ax.set_xlim(0.35, 10.5)
        ax.set_xlabel("Attributed CPU cores | log scale")
        ax.set_ylabel("End-to-end FPS")
        _panel_heading(ax, panel, theme)
        ax.text(
            0.99,
            0.03,
            "bubble area = peak VRAM",
            transform=ax.transAxes,
            color=theme.muted_text,
            fontsize=7.5,
            ha="right",
            va="bottom",
        )

    figure.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.09, hspace=0.42)
    _save_figure(figure, output_path, theme)


def render_lifecycle(
    data: PublishedFigureData,
    output_path: Path,
    theme: Theme,
) -> None:
    """Render full-process startup, steady-state, and finalize medians."""
    _configure_matplotlib(theme)
    figure, axes = plt.subplots(2, 2, figsize=(11.6, 7.6))
    for ax, panel in zip(axes.flat, data.panels, strict=True):
        _style_panel(ax, theme, horizontal_grid=False)
        results = [panel.result(name) for name in IMPLEMENTATION_ORDER]
        y_positions = list(reversed(range(len(results))))
        for y, result in zip(y_positions, results, strict=True):
            left = 0.0
            for value, color in (
                (result.startup_sec, theme.startup),
                (result.steady_state_sec, theme.steady),
                (result.finalize_sec, theme.finalize),
            ):
                ax.barh(
                    y,
                    value,
                    left=left,
                    height=0.48,
                    color=color,
                    edgecolor="none",
                )
                left += value
            ax.text(
                result.wall_sec * 1.015,
                y,
                (
                    f"{result.startup_sec:.2f}s / "
                    f"{result.steady_state_sec:.2f}s / "
                    f"{result.finalize_sec:.2f}s"
                ),
                color=theme.muted_text,
                fontsize=7.2,
                va="center",
            )

        maximum = max(result.wall_sec for result in results)
        ax.set_xlim(0, maximum * 1.47)
        ax.set_yticks(
            y_positions,
            [IMPLEMENTATION_LABELS[result.implementation] for result in results],
        )
        ax.set_xlabel("Full-process wall time | seconds")
        _panel_heading(ax, panel, theme)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in (theme.startup, theme.steady, theme.finalize)
    ]
    figure.legend(
        handles,
        ("startup", "steady-state", "finalize"),
        loc="lower center",
        ncol=3,
        frameon=False,
        labelcolor=theme.text,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.12, hspace=0.42)
    _save_figure(figure, output_path, theme)


def generate_figures(results_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    """Generate every committed benchmark figure from published JSON."""
    data = load_published_data(results_dir)
    generated: list[Path] = []
    renderers = {
        "tuned-sweep": render_tuned_sweep,
        "throughput-resources": render_throughput_resources,
        "lifecycle": render_lifecycle,
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
        help="Directory containing tuned.json and upstream-default.json",
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
