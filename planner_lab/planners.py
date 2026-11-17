from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .geometry import (
    Point,
    TimedPoint,
    angle_diff,
    angle_of,
    clamp,
    cubic_bezier,
    distance,
    normalize,
    path_smoothness,
    polyline_length,
    resample_timed,
    rotate,
)
from .scenario import Scenario, obstacle_snapshot, timed_point_to_dict


@dataclass
class CandidateTrajectory:
    path: list[TimedPoint]
    cost: float
    min_clearance: float
    collision_count: int
    components: dict[str, float] = field(default_factory=dict)

    @property
    def collides(self) -> bool:
        return self.collision_count > 0


@dataclass
class PlanResult:
    selected: CandidateTrajectory
    candidates: list[CandidateTrajectory]
    compute_time_ms: float
    next_heading: float | None = None


@dataclass
class StepSnapshot:
    step: int
    t: float
    current: TimedPoint
    selected: list[TimedPoint]
    candidates: list[list[TimedPoint]]
    obstacles: list[dict[str, float | str | bool]]
    compute_time_ms: float
    selected_cost: float
    selected_components: dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationResult:
    method: str
    executed: list[TimedPoint]
    snapshots: list[StepSnapshot]
    compute_times_ms: list[float]
    success: bool
    reached_at_t: float | None


class Planner:
    name = "base"

    def plan(self, current: Point, heading: float, t: float, scenario: Scenario) -> PlanResult:
        raise NotImplementedError

    def evaluate(self, path: list[TimedPoint], scenario: Scenario) -> CandidateTrajectory:
        min_clearance = float("inf")
        collision_count = 0
        clearance_penalty = 0.0
        for point in path:
            clearance = scenario.clearance(point.point, point.t)
            min_clearance = min(min_clearance, clearance)
            if clearance < 0.0:
                collision_count += 1
            if clearance < 0.65:
                clearance_penalty += (0.65 - clearance) ** 2

        length = polyline_length(path)
        smoothness = path_smoothness(path)
        goal_dist = distance(path[-1].point, scenario.goal)
        collision_penalty = 100000.0 * collision_count
        clearance_cost = 65.0 * clearance_penalty
        cost = (
            4.0 * goal_dist
            + 0.25 * length
            + 7.5 * smoothness
            + clearance_cost
            + collision_penalty
        )
        return CandidateTrajectory(
            path=path,
            cost=cost,
            min_clearance=min_clearance,
            collision_count=collision_count,
            components={
                "goal": goal_dist,
                "length": length,
                "smoothness": smoothness,
                "clearance_penalty": clearance_penalty,
                "collision_count": float(collision_count),
            },
        )


class SamplingPlanner(Planner):
    name = "sampling"

    def __init__(self, horizon_s: float = 2.8, dt: float = 0.2) -> None:
        self.horizon_s = horizon_s
        self.dt = dt
        self.steps = int(round(horizon_s / dt)) + 1

    def plan(self, current: Point, heading: float, t: float, scenario: Scenario) -> PlanResult:
        started = time.perf_counter()
        candidates: list[CandidateTrajectory] = []
        to_goal = scenario.goal - current
        goal_heading = angle_of(to_goal)
        base_heading = heading if distance(current, scenario.start) > 0.2 else goal_heading
        heading_error = angle_diff(goal_heading, base_heading)

        offsets_deg = [-75, -55, -38, -24, -12, 0, 12, 24, 38, 55, 75]
        speeds = [0.72, 0.96, 1.18, 1.42]
        if distance(current, scenario.goal) < 3.0:
            speeds = [0.45, 0.7, 0.95]
            offsets_deg = [-45, -25, -12, 0, 12, 25, 45]

        for speed in speeds:
            for offset_deg in offsets_deg:
                offset = math.radians(offset_deg) + 0.25 * heading_error
                travel_heading = base_heading + offset
                end = Point(
                    current.x + math.cos(travel_heading) * speed * self.horizon_s,
                    current.y + math.sin(travel_heading) * speed * self.horizon_s,
                )
                if distance(current, scenario.goal) < speed * self.horizon_s:
                    end = scenario.goal
                end = scenario.clamp_point(end)

                chord = max(distance(current, end), 0.2)
                start_dir = Point(math.cos(base_heading), math.sin(base_heading))
                end_dir = normalize(scenario.goal - end)
                if distance(end, scenario.goal) < 0.3:
                    end_dir = normalize(end - current)
                p1 = current + start_dir * (0.38 * chord)
                p2 = end - end_dir * (0.32 * chord)
                points = [
                    scenario.clamp_point(cubic_bezier(current, p1, p2, end, i / (self.steps - 1)))
                    for i in range(self.steps)
                ]
                path = resample_timed(points, t, self.dt)
                candidates.append(self.evaluate(path, scenario))

        candidates.sort(key=lambda item: item.cost)
        compute_time_ms = (time.perf_counter() - started) * 1000.0
        return PlanResult(candidates[0], candidates[:12], compute_time_ms)


class ElasticBandPlanner(Planner):
    name = "elastic_band"

    def __init__(self, horizon_s: float = 3.4, dt: float = 0.2, iterations: int = 45) -> None:
        self.horizon_s = horizon_s
        self.dt = dt
        self.steps = int(round(horizon_s / dt)) + 1
        self.iterations = iterations

    def plan(self, current: Point, heading: float, t: float, scenario: Scenario) -> PlanResult:
        started = time.perf_counter()
        candidates: list[CandidateTrajectory] = []
        goal_vec = scenario.goal - current
        goal_dist = distance(current, scenario.goal)
        max_reach = 1.08 * self.horizon_s
        if goal_dist <= max_reach:
            local_goal = scenario.goal
        else:
            local_goal = current + normalize(goal_vec) * max_reach

        amplitudes = [-2.0, -1.05, 0.0, 1.05, 2.0]
        direction = normalize(local_goal - current)
        if distance(current, local_goal) < 0.1:
            direction = Point(math.cos(heading), math.sin(heading))
        normal = rotate(direction, math.pi / 2.0)

        for amp in amplitudes:
            initial: list[Point] = []
            for i in range(self.steps):
                u = i / (self.steps - 1)
                base = Point(
                    current.x + (local_goal.x - current.x) * u,
                    current.y + (local_goal.y - current.y) * u,
                )
                bump = math.sin(math.pi * u) * amp
                initial.append(scenario.clamp_point(base + normal * bump))
            optimized = self._optimize(initial, t, scenario)
            candidates.append(self.evaluate(resample_timed(optimized, t, self.dt), scenario))

        candidates.sort(key=lambda item: item.cost)
        compute_time_ms = (time.perf_counter() - started) * 1000.0
        return PlanResult(candidates[0], candidates[:12], compute_time_ms)

    def _optimize(self, points: list[Point], start_t: float, scenario: Scenario) -> list[Point]:
        original = [Point(p.x, p.y) for p in points]
        band = [Point(p.x, p.y) for p in points]
        influence = 1.45
        step_size = 0.12

        for _ in range(self.iterations):
            new_band = [band[0]]
            for i in range(1, len(band) - 1):
                p = band[i]
                t = start_t + i * self.dt
                prev_p = band[i - 1]
                next_p = band[i + 1]
                midpoint = (prev_p + next_p) * 0.5
                smooth_force = (midpoint - p) * 1.25
                anchor_force = (original[i] - p) * 0.18
                repulse = Point(0.0, 0.0)

                for obstacle in scenario.obstacles:
                    center = obstacle.position_at(t)
                    away = p - center
                    d = max(distance(p, center), 1e-4)
                    clearance = d - obstacle.radius - scenario.robot_radius
                    if clearance < influence:
                        strength = ((influence - clearance) / influence) ** 2
                        repulse = repulse + (away / d) * (2.1 * strength)

                updated = p + (smooth_force + anchor_force + repulse) * step_size
                new_band.append(scenario.clamp_point(updated))
            new_band.append(band[-1])
            band = new_band
        return band


def simulate(
    planner: Planner,
    scenario: Scenario,
    max_steps: int = 95,
    step_dt: float = 0.2,
    goal_tolerance: float = 0.42,
) -> SimulationResult:
    current = scenario.start
    heading = angle_of(scenario.goal - scenario.start)
    t = 0.0
    executed = [TimedPoint(current.x, current.y, t)]
    snapshots: list[StepSnapshot] = []
    compute_times: list[float] = []
    reached_at: float | None = None

    for step in range(max_steps):
        if distance(current, scenario.goal) <= goal_tolerance:
            reached_at = t
            break

        result = planner.plan(current, heading, t, scenario)
        selected = result.selected.path
        next_index = 1 if len(selected) > 1 else 0
        next_point = selected[next_index].point
        if result.next_heading is not None:
            heading = result.next_heading
        elif distance(next_point, current) > 1e-6:
            heading = angle_of(next_point - current)
        current = next_point
        t += step_dt
        executed.append(TimedPoint(current.x, current.y, t))
        compute_times.append(result.compute_time_ms)
        snapshots.append(
            StepSnapshot(
                step=step,
                t=t,
                current=TimedPoint(current.x, current.y, t),
                selected=selected,
                candidates=[candidate.path for candidate in result.candidates],
                obstacles=obstacle_snapshot(scenario, t),
                compute_time_ms=result.compute_time_ms,
                selected_cost=result.selected.cost,
                selected_components=result.selected.components,
            )
        )

    success = reached_at is not None or distance(current, scenario.goal) <= goal_tolerance
    if success and reached_at is None:
        reached_at = t
    return SimulationResult(
        method=planner.name,
        executed=executed,
        snapshots=snapshots,
        compute_times_ms=compute_times,
        success=success,
        reached_at_t=reached_at,
    )


def simulation_to_dict(result: SimulationResult) -> dict:
    return {
        "method": result.method,
        "executed": [timed_point_to_dict(point) for point in result.executed],
        "snapshots": [
            {
                "step": snapshot.step,
                "t": snapshot.t,
                "current": timed_point_to_dict(snapshot.current),
                "selected": [timed_point_to_dict(point) for point in snapshot.selected],
                "candidates": [
                    [timed_point_to_dict(point) for point in candidate]
                    for candidate in snapshot.candidates
                ],
                "obstacles": snapshot.obstacles,
                "compute_time_ms": snapshot.compute_time_ms,
                "selected_cost": snapshot.selected_cost,
                "selected_components": snapshot.selected_components,
            }
            for snapshot in result.snapshots
        ],
        "compute_times_ms": result.compute_times_ms,
        "success": result.success,
        "reached_at_t": result.reached_at_t,
    }
