from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .geometry import Point, distance, path_smoothness, polyline_length
from .planners import SimulationResult
from .scenario import Scenario


@dataclass(frozen=True)
class Metrics:
    method: str
    success: bool
    reached_at_t: float | None
    elapsed_time_s: float
    final_goal_distance: float
    min_goal_distance: float
    trajectory_length: float
    smoothness: float
    min_safety_distance: float
    collision_rate: float
    collision_steps: int
    total_steps: int
    average_speed: float
    max_collision_probability: float
    avg_compute_time_ms: float
    max_compute_time_ms: float

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        return {
            "method": self.method,
            "success": self.success,
            "reached_at_t": self.reached_at_t,
            "elapsed_time_s": self.elapsed_time_s,
            "final_goal_distance": self.final_goal_distance,
            "min_goal_distance": self.min_goal_distance,
            "trajectory_length": self.trajectory_length,
            "smoothness": self.smoothness,
            "min_safety_distance": self.min_safety_distance,
            "collision_rate": self.collision_rate,
            "collision_steps": self.collision_steps,
            "total_steps": self.total_steps,
            "average_speed": self.average_speed,
            "max_collision_probability": self.max_collision_probability,
            "avg_compute_time_ms": self.avg_compute_time_ms,
            "max_compute_time_ms": self.max_compute_time_ms,
        }


def compute_metrics(result: SimulationResult, scenario: Scenario) -> Metrics:
    clearances = [scenario.clearance(point.point, point.t) for point in result.executed]
    collision_steps = sum(1 for clearance in clearances if clearance < 0.0)
    total_steps = len(result.executed)
    times = result.compute_times_ms
    elapsed = result.executed[-1].t - result.executed[0].t if len(result.executed) > 1 else 0.0
    goal_distances = [distance(point.point, scenario.goal) for point in result.executed]
    trajectory_length = polyline_length(result.executed)
    return Metrics(
        method=result.method,
        success=result.success,
        reached_at_t=result.reached_at_t,
        elapsed_time_s=elapsed,
        final_goal_distance=goal_distances[-1] if goal_distances else float("inf"),
        min_goal_distance=min(goal_distances) if goal_distances else float("inf"),
        trajectory_length=trajectory_length,
        smoothness=path_smoothness(result.executed),
        min_safety_distance=min(clearances) if clearances else float("inf"),
        collision_rate=collision_steps / max(1, total_steps),
        collision_steps=collision_steps,
        total_steps=total_steps,
        average_speed=trajectory_length / elapsed if elapsed > 1e-9 else 0.0,
        max_collision_probability=estimate_executed_max_cp(result, scenario),
        avg_compute_time_ms=sum(times) / max(1, len(times)),
        max_compute_time_ms=max(times) if times else 0.0,
    )


def estimate_executed_max_cp(result: SimulationResult, scenario: Scenario, samples: int = 48) -> float:
    if not scenario.dynamic_obstacles:
        return 0.0
    rng = random.Random(911)
    normal_samples = [(rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)) for _ in range(samples)]
    max_cp = 0.0
    for point in result.executed:
        max_cp = max(max_cp, estimate_point_joint_cp(point.point, point.t, scenario, normal_samples))
    return max_cp


def estimate_point_joint_cp(
    point: Point,
    t: float,
    scenario: Scenario,
    normal_samples: list[tuple[float, float]],
) -> float:
    no_collision = 1.0
    for obstacle in scenario.dynamic_obstacles:
        sigma = obstacle.prediction_radial_sigma_at(point, t)
        mean = obstacle.predicted_position_at(t)
        threshold = obstacle.radius + scenario.robot_radius
        if sigma <= 1e-6:
            cp = 1.0 if distance(point, mean) < threshold else 0.0
        else:
            z = (threshold - distance(point, mean)) / max(1e-6, sigma)
            analytic = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            hits = 0
            for zx, zy in normal_samples:
                sample = obstacle.prediction_sample_at(t, zx, zy)
                if distance(point, sample) < threshold:
                    hits += 1
            monte_carlo = hits / len(normal_samples)
            cp = max(0.0, min(1.0, 0.65 * analytic + 0.35 * monte_carlo))
        no_collision *= 1.0 - cp
    return 1.0 - no_collision
