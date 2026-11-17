from __future__ import annotations

import html
import math
from pathlib import Path

from .geometry import Point, TimedPoint
from .metrics import Metrics
from .planners import SimulationResult, StepSnapshot
from .scenario import GaussianPrediction, Scenario


Palette = {
    "sampling": "#2563eb",
    "elastic_band": "#c2410c",
    "vanilla_mppi": "#2563eb",
    "risk_aware_mppi": "#dc2626",
    "static": "#475569",
    "dynamic": "#be123c",
    "candidate": "#94a3b8",
    "selected": "#0891b2",
    "executed": "#111827",
    "grid": "#e2e8f0",
    "safe": "#f1f5f9",
}


class SvgCanvas:
    def __init__(self, scenario: Scenario, width_units: float | None = None, height_units: float | None = None) -> None:
        self.scenario = scenario
        self.margin = 34.0
        self.scale = 42.0
        self.width_units = width_units or scenario.width
        self.height_units = height_units or scenario.height
        self.width = int(self.margin * 2 + self.width_units * self.scale)
        self.height = int(self.margin * 2 + self.height_units * self.scale)

    def xy(self, point: Point | TimedPoint) -> tuple[float, float]:
        p = point.point if isinstance(point, TimedPoint) else point
        return self.margin + p.x * self.scale, self.margin + (self.scenario.height - p.y) * self.scale

    def point_string(self, points: list[Point] | list[TimedPoint]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in (self.xy(point) for point in points))


def _path(points: list[Point] | list[TimedPoint], color: str, width: float, opacity: float = 1.0, dash: str = "") -> str:
    if len(points) < 2:
        return ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{_active_canvas.point_string(points)}" fill="none" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
        f'stroke-linejoin="round" opacity="{opacity}"{dash_attr}/>'
    )


def _circle(point: Point, radius: float, color: str, opacity: float, stroke: str = "none") -> str:
    x, y = _active_canvas.xy(point)
    stroke_attr = f' stroke="{stroke}" stroke-width="1.2"' if stroke != "none" else ""
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius * _active_canvas.scale:.1f}" '
        f'fill="{color}" opacity="{opacity}"{stroke_attr}/>'
    )


def _circle_outline(point: Point, radius: float, color: str, opacity: float, dash: str = "") -> str:
    x, y = _active_canvas.xy(point)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius * _active_canvas.scale:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="1.4" opacity="{opacity}"{dash_attr}/>'
    )


def _prediction_ellipse(
    prediction: GaussianPrediction,
    radius: float,
    sigma_scale: float,
    color: str,
    opacity: float,
    dash: str = "",
) -> str:
    x, y = _active_canvas.xy(prediction.mean)
    rx = (radius + sigma_scale * prediction.sigma_longitudinal) * _active_canvas.scale
    ry = (radius + sigma_scale * prediction.sigma_lateral) * _active_canvas.scale
    angle_degrees = -math.degrees(prediction.angle)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
        f'transform="rotate({angle_degrees:.1f} {x:.1f} {y:.1f})" fill="none" '
        f'stroke="{color}" stroke-width="1.4" opacity="{opacity}"{dash_attr}/>'
    )


def _label(point: Point, text: str, color: str = "#111827", dy: float = -9.0) -> str:
    x, y = _active_canvas.xy(point)
    return (
        f'<text x="{x:.1f}" y="{y + dy:.1f}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="11" fill="{color}">'
        f"{html.escape(text)}</text>"
    )


def _base_scene(
    scenario: Scenario,
    t: float,
    show_dynamic: bool = True,
    goal_tolerance: float | None = None,
) -> list[str]:
    parts: list[str] = []
    parts.append(
        f'<rect x="{_active_canvas.margin:.1f}" y="{_active_canvas.margin:.1f}" '
        f'width="{scenario.width * _active_canvas.scale:.1f}" height="{scenario.height * _active_canvas.scale:.1f}" '
        'fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2"/>'
    )
    for gx in range(int(scenario.width) + 1):
        x = _active_canvas.margin + gx * _active_canvas.scale
        parts.append(
            f'<line x1="{x:.1f}" y1="{_active_canvas.margin:.1f}" '
            f'x2="{x:.1f}" y2="{_active_canvas.margin + scenario.height * _active_canvas.scale:.1f}" '
            f'stroke="{Palette["grid"]}" stroke-width="0.6"/>'
        )
    for gy in range(int(scenario.height) + 1):
        y = _active_canvas.margin + gy * _active_canvas.scale
        parts.append(
            f'<line x1="{_active_canvas.margin:.1f}" y1="{y:.1f}" '
            f'x2="{_active_canvas.margin + scenario.width * _active_canvas.scale:.1f}" y2="{y:.1f}" '
            f'stroke="{Palette["grid"]}" stroke-width="0.6"/>'
        )

    for obstacle in scenario.static_obstacles:
        parts.append(_circle(obstacle.center, obstacle.radius + scenario.robot_radius, Palette["safe"], 1.0, "#cbd5e1"))
        parts.append(_circle(obstacle.center, obstacle.radius, Palette["static"], 0.72))
        parts.append(_label(obstacle.center, obstacle.name, "#334155", dy=4.0))

    if show_dynamic:
        for obstacle in scenario.dynamic_obstacles:
            pos = obstacle.position_at(t)
            pred = obstacle.predicted_position_at(t)
            sigma = obstacle.prediction_sigma_at(t)
            trail = [obstacle.position_at(max(0.0, t - i * 0.8)) for i in range(5, -1, -1)]
            parts.append(_path(trail, Palette["dynamic"], 1.2, 0.35, "3 4"))
            if sigma > 1e-6:
                parts.append(_circle_outline(pred, obstacle.radius + scenario.robot_radius + 2.0 * sigma, "#fb7185", 0.48, "5 4"))
                parts.append(_circle(pred, 0.055, "#fb7185", 0.8))
            parts.append(_circle(pos, obstacle.radius + scenario.robot_radius, "#ffe4e6", 1.0, "#fecdd3"))
            parts.append(_circle(pos, obstacle.radius, Palette["dynamic"], 0.72))
            parts.append(_label(pos, obstacle.name, "#881337", dy=4.0))

    parts.append(_circle(scenario.start, 0.17, "#16a34a", 1.0))
    if goal_tolerance is not None:
        parts.append(_circle_outline(scenario.goal, goal_tolerance, "#7c3aed", 0.42, "6 4"))
    parts.append(_circle(scenario.goal, 0.2, "#7c3aed", 1.0))
    parts.append(_label(scenario.start, "start", "#166534", dy=-10.0))
    parts.append(_label(scenario.goal, "goal", "#581c87", dy=-10.0))
    return parts


def render_comparison(
    path: Path,
    scenario: Scenario,
    results: list[SimulationResult],
    metrics: list[Metrics],
    goal_tolerance: float | None = None,
) -> None:
    global _active_canvas
    _active_canvas = SvgCanvas(scenario)
    table_height = _metric_table_height(metrics)
    svg_width = max(_active_canvas.width, 1040)
    parts = _svg_open(svg_width, _active_canvas.height + table_height)
    parts.extend(_base_scene(scenario, t=0.0, goal_tolerance=goal_tolerance))
    for result in results:
        color = Palette.get(result.method, Palette["executed"])
        parts.append(_path(result.executed, color, 3.4, 0.94))
        if result.executed:
            parts.append(_circle(result.executed[-1].point, 0.13, "#ffffff", 1.0, color))
            label = "reached" if result.success else "stopped"
            parts.append(_label(result.executed[-1].point, label, color, dy=16.0))

    legend_y = _active_canvas.height + 8
    parts.extend(_metric_table(metrics, legend_y, "Final executed trajectories"))

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def render_process_strip(
    path: Path,
    scenario: Scenario,
    result: SimulationResult,
    panels: int = 6,
    goal_tolerance: float | None = None,
) -> None:
    global _active_canvas
    single = SvgCanvas(scenario)
    gap = 18
    label_h = 42
    width = single.width * panels + gap * (panels - 1)
    height = single.height + label_h
    parts = _svg_open(width, height)
    snapshots = _sample_snapshots(result.snapshots, panels)

    for i, snapshot in enumerate(snapshots):
        x_offset = i * (single.width + gap)
        parts.append(f'<g transform="translate({x_offset}, {label_h})">')
        _active_canvas = single
        parts.extend(_base_scene(scenario, snapshot.t, goal_tolerance=goal_tolerance))
        for candidate in snapshot.candidates:
            parts.append(_path(candidate, Palette["candidate"], 1.2, 0.35))
        parts.append(_path(snapshot.selected, Palette["selected"], 3.2, 0.95))
        prefix = [p for p in result.executed if p.t <= snapshot.t + 1e-9]
        parts.append(_path(prefix, Palette.get(result.method, Palette["executed"]), 3.0, 0.95))
        parts.append(_circle(snapshot.current.point, 0.14, "#111827", 1.0))
        parts.append("</g>")
        title = f"{result.method} step {snapshot.step}, t={snapshot.t:.1f}s"
        parts.append(_text(x_offset + 8, 18, title, 13, "#111827", weight="700"))
        parts.append(_text(x_offset + 8, 36, f"cost={snapshot.selected_cost:.1f}, compute={snapshot.compute_time_ms:.2f} ms", 11, "#475569"))

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def render_animation(
    path: Path,
    scenario: Scenario,
    results: list[SimulationResult],
    metrics: list[Metrics],
    goal_tolerance: float | None = None,
) -> None:
    global _active_canvas
    _active_canvas = SvgCanvas(scenario)
    duration = max((result.executed[-1].t for result in results if result.executed), default=1.0)
    frame_count = 80
    times = [duration * i / max(1, frame_count - 1) for i in range(frame_count)]
    table_height = _metric_table_height(metrics, subtitle_lines=1)
    svg_width = max(_active_canvas.width, 1040)
    parts = _svg_open(svg_width, _active_canvas.height + table_height)
    parts.extend(_base_scene(scenario, t=0.0, show_dynamic=False, goal_tolerance=goal_tolerance))

    for obstacle in scenario.dynamic_obstacles:
        true_points = [obstacle.position_at(t) for t in times]
        pred_points = [obstacle.predicted_position_at(t) for t in times]
        risk_radii = [
            obstacle.radius + scenario.robot_radius + 2.0 * obstacle.prediction_sigma_at(t)
            for t in times
        ]
        parts.append(_path(true_points, Palette["dynamic"], 1.2, 0.35, "3 4"))
        parts.append(_animated_circle(pred_points, risk_radii, duration, "#fb7185", 0.32, fill="none", dash="5 4"))
        parts.append(_animated_circle(true_points, [obstacle.radius + scenario.robot_radius] * frame_count, duration, "#ffe4e6", 0.9, stroke="#fecdd3"))
        parts.append(_animated_circle(true_points, [obstacle.radius] * frame_count, duration, Palette["dynamic"], 0.74))
        parts.append(_animated_label(true_points, obstacle.name, duration, "#881337"))

    for result in results:
        color = Palette.get(result.method, Palette["executed"])
        parts.append(_path(result.executed, color, 2.0, 0.25))
        sampled = _path_at_times(result.executed, times)
        parts.append(_animated_circle(sampled, [0.18] * frame_count, duration, color, 1.0, stroke="#111827"))

    legend_y = _active_canvas.height + 8
    parts.extend(
        _metric_table(
            metrics,
            legend_y,
            "Animated closed-loop execution",
            "Open this SVG in a browser or image viewer that supports SVG animation.",
        )
    )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def render_prediction_ellipses(
    path: Path,
    scenario: Scenario,
    times: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0),
) -> None:
    global _active_canvas
    single = SvgCanvas(scenario)
    gap = 18
    label_h = 48
    legend_h = 54
    width = single.width * len(times) + gap * max(0, len(times) - 1)
    height = single.height + label_h + legend_h
    parts = _svg_open(width, height)
    collision_radius = scenario.robot_radius

    for index, t in enumerate(times):
        x_offset = index * (single.width + gap)
        parts.append(f'<g transform="translate({x_offset}, {label_h})">')
        _active_canvas = single
        parts.extend(_base_scene(scenario, t, show_dynamic=False))
        for obstacle in scenario.dynamic_obstacles:
            prediction = obstacle.prediction_at(t)
            actual = obstacle.position_at(t)
            envelope_radius = obstacle.radius + collision_radius
            parts.append(
                _prediction_ellipse(
                    prediction,
                    envelope_radius,
                    2.0,
                    "#be123c",
                    0.72,
                    "6 4",
                )
            )
            parts.append(
                _prediction_ellipse(
                    prediction,
                    envelope_radius,
                    1.0,
                    "#fb7185",
                    0.88,
                )
            )
            parts.append(_circle(prediction.mean, 0.06, "#fb7185", 0.95))
            parts.append(_circle(actual, obstacle.radius, Palette["dynamic"], 0.76))
            parts.append(_label(actual, obstacle.name, "#881337", dy=4.0))
        parts.append("</g>")
        parts.append(_text(x_offset + 10, 20, f"prediction t={t:.1f}s", 13, "#111827", weight="700"))
        parts.append(
            _text(
                x_offset + 10,
                38,
                "ellipse axes follow obstacle velocity",
                11,
                "#475569",
            )
        )

    legend_y = label_h + single.height + 20
    parts.append(_text(34, legend_y, "solid: 1 sigma collision envelope", 12, "#be123c", weight="700"))
    parts.append(_text(290, legend_y, "dashed: 2 sigma collision envelope", 12, "#be123c", weight="700"))
    parts.append(
        _text(
            34,
            legend_y + 22,
            "Ellipse radii include dynamic-obstacle and robot collision radii.",
            11,
            "#475569",
        )
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _animated_circle(
    points: list[Point],
    radii: list[float],
    duration: float,
    color: str,
    opacity: float,
    fill: str | None = None,
    stroke: str = "none",
    dash: str = "",
) -> str:
    xs = []
    ys = []
    for point in points:
        x, y = _active_canvas.xy(point)
        xs.append(x)
        ys.append(y)
    fill_value = fill if fill is not None else color
    stroke_attr = f' stroke="{stroke}" stroke-width="1.2"' if stroke != "none" else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<circle cx="{xs[0]:.1f}" cy="{ys[0]:.1f}" r="{radii[0] * _active_canvas.scale:.1f}" '
        f'fill="{fill_value}" opacity="{opacity}"{stroke_attr}{dash_attr}>'
        f'<animate attributeName="cx" values="{_number_values(xs)}" dur="{duration:.2f}s" repeatCount="indefinite"/>'
        f'<animate attributeName="cy" values="{_number_values(ys)}" dur="{duration:.2f}s" repeatCount="indefinite"/>'
        f'<animate attributeName="r" values="{_number_values([r * _active_canvas.scale for r in radii])}" dur="{duration:.2f}s" repeatCount="indefinite"/>'
        '</circle>'
    )


def _animated_label(points: list[Point], text: str, duration: float, color: str) -> str:
    xs = []
    ys = []
    for point in points:
        x, y = _active_canvas.xy(point)
        xs.append(x)
        ys.append(y + 4.0)
    return (
        f'<text x="{xs[0]:.1f}" y="{ys[0]:.1f}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="11" fill="{color}">{html.escape(text)}'
        f'<animate attributeName="x" values="{_number_values(xs)}" dur="{duration:.2f}s" repeatCount="indefinite"/>'
        f'<animate attributeName="y" values="{_number_values(ys)}" dur="{duration:.2f}s" repeatCount="indefinite"/>'
        '</text>'
    )


def _path_at_times(path: list[TimedPoint], times: list[float]) -> list[Point]:
    if not path:
        return [Point(0.0, 0.0) for _ in times]
    points: list[Point] = []
    j = 0
    for t in times:
        while j + 1 < len(path) and path[j + 1].t < t:
            j += 1
        if j + 1 >= len(path):
            points.append(path[-1].point)
            continue
        a = path[j]
        b = path[j + 1]
        if b.t <= a.t:
            points.append(a.point)
            continue
        u = max(0.0, min(1.0, (t - a.t) / (b.t - a.t)))
        points.append(Point(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u))
    return points


def _number_values(values: list[float]) -> str:
    return ";".join(f"{value:.1f}" for value in values)


def _sample_snapshots(snapshots: list[StepSnapshot], count: int) -> list[StepSnapshot]:
    if len(snapshots) <= count:
        return snapshots
    indexes = [round(i * (len(snapshots) - 1) / (count - 1)) for i in range(count)]
    return [snapshots[i] for i in indexes]


def _format_reached(row: Metrics) -> str:
    if row.reached_at_t is None:
        return "not reached"
    return f"{row.reached_at_t:.1f}s"


def _metric_table_height(metrics: list[Metrics], subtitle_lines: int = 0) -> int:
    header_height = 70 + subtitle_lines * 18
    return max(128, header_height + 22 * len(metrics))


def _metric_table(metrics: list[Metrics], y: float, title: str, subtitle: str | None = None) -> list[str]:
    parts = [_text(34, y, title, 15, "#111827", weight="700")]
    header_y = y + 24
    if subtitle is not None:
        parts.append(_text(34, y + 20, subtitle, 11, "#475569"))
        header_y += 18

    columns = [
        (76, "method"),
        (210, "success"),
        (282, "reached"),
        (362, "length"),
        (440, "smooth"),
        (532, "min_clear"),
        (638, "collision"),
        (746, "max_cp"),
        (846, "avg_time"),
    ]
    for x, label in columns:
        parts.append(_text(x, header_y, label, 11, "#475569", weight="700"))

    row_y = header_y + 22
    for row in metrics:
        color = Palette.get(row.method, Palette["executed"])
        parts.append(f'<line x1="36" y1="{row_y - 5}" x2="66" y2="{row_y - 5}" stroke="{color}" stroke-width="3.5"/>')
        values = [
            row.method,
            str(row.success),
            _format_reached(row),
            f"{row.trajectory_length:.2f}",
            f"{row.smoothness:.4f}",
            f"{row.min_safety_distance:.2f}",
            f"{row.collision_rate:.3f}",
            f"{row.max_collision_probability:.2f}",
            f"{row.avg_compute_time_ms:.2f} ms",
        ]
        for (x, _), value in zip(columns, values):
            parts.append(_text(x, row_y, value, 11, "#334155"))
        row_y += 22
    return parts


def _svg_open(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
    ]


def _text(x: float, y: float, text: str, size: int, color: str, weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">{html.escape(text)}</text>'
    )


_active_canvas: SvgCanvas
