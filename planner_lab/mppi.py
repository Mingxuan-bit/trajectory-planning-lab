from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from .geometry import Point, TimedPoint, angle_diff, angle_of, clamp, distance, dot, norm, path_smoothness, polyline_length, rotate
from .planners import CandidateTrajectory, PlanResult, Planner
from .scenario import DynamicCircleObstacle, DynamicObstacle, GaussianPrediction, Scenario


CHANCE_SAFETY_FALLBACKS = ("least_violation", "brake")
# CandidateTrajectory.components is numeric; human-readable labels belong in traces.
SELECTION_REASON_LABELS = {
    0.0: "cost_selection",
    1.0: "feasible_selection",
    2.0: "least_violation",
    3.0: "explicit_brake",
}


@dataclass(frozen=True)
class Control:
    v: float
    steer: float


@dataclass(frozen=True)
class PredictionStep:
    actual_centers: tuple[Point, ...]
    predictions: tuple[GaussianPrediction, ...]


class MPPIPlanner(Planner):
    def __init__(
        self,
        name: str,
        risk_aware: bool,
        seed: int = 0,
        horizon_steps: int = 24,
        dt: float = 0.18,
        rollouts: int = 220,
        mc_samples: int = 24,
        temperature: float = 4.5,
        lambda_cp: float = 60.0,
        lambda_hard: float = 180.0,
        cp_threshold: float = 0.1,
        enforce_chance_constraint: bool = False,
        chance_constraint_steps: int | None = None,
        chance_safety_fallback: str = "least_violation",
        uncertainty_margin_weight: float = 120.0,
        uncertainty_margin_sigma: float = 2.0,
        obstacle_inflation_sigma: float = 0.0,
        covariance_mode: str = "model",
        wheel_base: float = 0.7,
        v_max: float = 1.55,
        a_max: float = 1.2,
        steer_max: float = 0.62,
        steer_rate_max: float = 1.6,
        min_turn_radius: float = 1.1,
        v_ref: float = 1.1,
        goal_capture_radius: float = 0.65,
    ) -> None:
        self.name = name
        self.risk_aware = risk_aware
        self.horizon_steps = horizon_steps
        self.dt = dt
        self.rollouts = rollouts
        self.mc_samples = mc_samples
        self.temperature = temperature
        self.lambda_cp = lambda_cp
        self.lambda_hard = lambda_hard
        self.cp_threshold = cp_threshold
        self.enforce_chance_constraint = enforce_chance_constraint
        if chance_safety_fallback not in CHANCE_SAFETY_FALLBACKS:
            raise ValueError(
                f"Unknown chance safety fallback: {chance_safety_fallback}. "
                f"Available: {', '.join(CHANCE_SAFETY_FALLBACKS)}"
            )
        self.chance_safety_fallback = chance_safety_fallback
        self.chance_constraint_steps = (
            horizon_steps
            if chance_constraint_steps is None
            else max(0, min(horizon_steps, chance_constraint_steps))
        )
        self.uncertainty_margin_weight = uncertainty_margin_weight
        self.uncertainty_margin_sigma = uncertainty_margin_sigma
        self.obstacle_inflation_sigma = obstacle_inflation_sigma
        self.covariance_mode = covariance_mode
        self.wheel_base = wheel_base
        self.v_min = 0.0
        self.v_max = v_max
        self.a_max = a_max
        self.min_turn_radius = min_turn_radius
        self.goal_capture_radius = goal_capture_radius
        self.steer_max = min(steer_max, math.atan(wheel_base / min_turn_radius))
        self.steer_rate_max = steer_rate_max
        self.v_noise = 0.32
        self.steer_noise = 0.24
        self.v_ref = min(v_ref, v_max)
        self.rng = random.Random(seed)
        self.nominal = [Control(0.7, 0.0) for _ in range(horizon_steps)]
        self.prev_control = Control(0.7, 0.0)
        self.normal_samples = [
            (self.rng.gauss(0.0, 1.0), self.rng.gauss(0.0, 1.0))
            for _ in range(max(1, mc_samples))
        ]

    def plan(self, current: Point, heading: float, t: float, scenario: Scenario) -> PlanResult:
        started = time.perf_counter()
        rollouts = self._sample_rollouts(current, heading, scenario)
        prediction_horizon = self._build_prediction_horizon(t, scenario)
        candidates: list[CandidateTrajectory] = []
        costs: list[float] = []

        for controls in rollouts:
            path = self._rollout_path(current, heading, t, controls, scenario)
            candidate = self._evaluate_mppi(path, controls, scenario, prediction_horizon)
            candidates.append(candidate)
            costs.append(candidate.cost)

        weights = self._weights(costs)
        updated = self._weighted_controls(rollouts, weights)
        weighted_path = self._rollout_path(current, heading, t, updated, scenario)
        weighted_candidate = self._evaluate_mppi(
            weighted_path,
            updated,
            scenario,
            prediction_horizon,
        )
        options = [*zip(candidates, rollouts), (weighted_candidate, updated)]
        static_safe_options = [
            option
            for option in options
            if self._static_feasible(option[0])
        ]
        selection_options = static_safe_options or options
        chance_safety_fallback_used = False
        selection_reason_code = 0.0
        if self.enforce_chance_constraint:
            feasible_options = [
                option
                for option in selection_options
                if self._chance_feasible(option[0])
            ]
            if feasible_options:
                selected, selected_controls = min(feasible_options, key=lambda option: option[0].cost)
                selection_reason_code = 1.0
            else:
                (
                    selected,
                    selected_controls,
                    chance_safety_fallback_used,
                ) = self._select_chance_safety_fallback(
                    current,
                    heading,
                    t,
                    scenario,
                    prediction_horizon,
                    selection_options,
                )
                selection_reason_code = 3.0 if chance_safety_fallback_used else 2.0
        else:
            selected, selected_controls = min(selection_options, key=lambda option: option[0].cost)

        selected.components["chance_safety_fallback_used"] = (
            1.0 if chance_safety_fallback_used else 0.0
        )
        selected.components["selection_reason_code"] = selection_reason_code
        selected.components["static_fallback_to_all"] = 0.0 if static_safe_options else 1.0
        self.prev_control = selected_controls[0]
        self.nominal = self._limit_control_sequence(
            selected_controls[1:] + [self._goal_seek_tail_control(selected.path, scenario)]
        )
        candidates.sort(key=lambda item: item.cost)
        display_candidates = [selected, *candidates[:11]]
        compute_time_ms = (time.perf_counter() - started) * 1000.0
        return PlanResult(
            selected,
            display_candidates,
            compute_time_ms,
            next_heading=self._next_heading(heading, selected_controls[0]),
        )

    def _select_chance_safety_fallback(
        self,
        current: Point,
        heading: float,
        t: float,
        scenario: Scenario,
        prediction_horizon: tuple[PredictionStep, ...],
        selection_options: list[tuple[CandidateTrajectory, list[Control]]],
    ) -> tuple[CandidateTrajectory, list[Control], bool]:
        if self.chance_safety_fallback == "brake":
            brake_controls = self._brake_rollout()
            brake_path = self._rollout_path(current, heading, t, brake_controls, scenario)
            brake_candidate = self._evaluate_mppi(
                brake_path,
                brake_controls,
                scenario,
                prediction_horizon,
            )
            if self._static_feasible(brake_candidate):
                return brake_candidate, brake_controls, True

        candidate, controls = min(
            selection_options,
            key=lambda option: (self._chance_violation(option[0]), option[0].cost),
        )
        return candidate, controls, False

    def _sample_rollouts(self, current: Point, heading: float, scenario: Scenario) -> list[list[Control]]:
        rollouts: list[list[Control]] = [self._limit_control_sequence(self.nominal)]
        for controls in self._guided_rollouts(current, heading, scenario):
            if len(rollouts) >= self.rollouts:
                return rollouts
            rollouts.append(controls)
        goal_dist = distance(current, scenario.goal)
        noise_scale = clamp(goal_dist / 4.0, 0.12, 1.0)
        for _ in range(self.rollouts - len(rollouts)):
            controls: list[Control] = []
            prev = self.prev_control
            for base in self.nominal:
                raw = Control(
                    base.v + self.rng.gauss(0.0, self.v_noise * noise_scale),
                    base.steer + self.rng.gauss(0.0, self.steer_noise * noise_scale),
                )
                control = self._limit_control(raw, prev)
                controls.append(control)
                prev = control
            rollouts.append(controls)
        return rollouts

    def _guided_rollouts(self, current: Point, heading: float, scenario: Scenario) -> list[list[Control]]:
        goal_vec = scenario.goal - current
        goal_dist = distance(current, scenario.goal)
        if goal_dist <= 1e-6:
            return [self._brake_rollout()]

        direction = goal_vec / goal_dist
        normal = rotate(direction, math.pi / 2.0)
        lateral_offsets = [0.0, -0.8, 0.8, -1.6, 1.6, -2.6, 2.6]
        if goal_dist < 2.4:
            lateral_offsets = [0.0, -0.45, 0.45]

        guided: list[list[Control]] = [self._brake_rollout(), self._goal_capture_rollout(current, heading, scenario)]
        for offset in lateral_offsets:
            target = scenario.clamp_point(scenario.goal + normal * offset)
            guided.append(self._pure_pursuit_rollout(current, heading, target, scenario.goal, scenario))
        if scenario.recovery_mode:
            guided.extend(self._recovery_rollouts())
        return guided

    def _recovery_rollouts(self) -> list[list[Control]]:
        rollouts: list[list[Control]] = []
        recovery_speeds = (
            max(0.18, min(self.v_ref, 0.42 * self.v_ref)),
            max(0.22, min(self.v_ref, 0.58 * self.v_ref)),
            -max(0.16, min(0.45, 0.34 * self.v_ref)),
            -max(0.20, min(0.55, 0.46 * self.v_ref)),
        )
        recovery_steers = (-self.steer_max, self.steer_max, -0.65 * self.steer_max, 0.65 * self.steer_max)
        for speed in recovery_speeds:
            v_min = min(self.v_min, speed)
            for steer in recovery_steers:
                controls: list[Control] = []
                prev = self.prev_control
                for step in range(self.horizon_steps):
                    turn_phase = step < max(2, self.horizon_steps // 2)
                    raw = Control(speed, steer if turn_phase else 0.35 * steer)
                    control = self._limit_control_with_min(raw, prev, v_min)
                    controls.append(control)
                    prev = control
                rollouts.append(controls)
        reverse_speeds = (
            -max(0.18, min(0.50, 0.42 * self.v_ref)),
            -max(0.22, min(0.60, 0.52 * self.v_ref)),
        )
        escape_speed = max(0.24, min(self.v_ref, 0.68 * self.v_ref))
        reverse_steps = max(5, self.horizon_steps // 2)
        straighten_steps = max(reverse_steps + 2, (2 * self.horizon_steps) // 3)
        for reverse_speed in reverse_speeds:
            v_min = min(self.v_min, reverse_speed)
            for steer in recovery_steers:
                controls = []
                prev = self.prev_control
                for step in range(self.horizon_steps):
                    if step < reverse_steps:
                        raw = Control(reverse_speed, steer)
                    elif step < straighten_steps:
                        raw = Control(escape_speed, -0.55 * steer)
                    else:
                        raw = Control(escape_speed, -0.20 * steer)
                    control = self._limit_control_with_min(raw, prev, v_min)
                    controls.append(control)
                    prev = control
                rollouts.append(controls)
        return rollouts

    def _brake_rollout(self) -> list[Control]:
        controls: list[Control] = []
        prev = self.prev_control
        for _ in range(self.horizon_steps):
            raw = Control(0.0, 0.65 * prev.steer)
            control = self._limit_control(raw, prev)
            controls.append(control)
            prev = control
        return controls

    def _pure_pursuit_rollout(
        self,
        current: Point,
        heading: float,
        target: Point,
        final_goal: Point,
        scenario: Scenario,
    ) -> list[Control]:
        x = current.x
        y = current.y
        theta = heading
        prev = self.prev_control
        controls: list[Control] = []
        for _ in range(self.horizon_steps):
            point = Point(x, y)
            goal_dist = distance(point, final_goal)
            if goal_dist <= self.goal_capture_radius:
                raw = Control(0.0, 0.35 * prev.steer)
            else:
                active_target = final_goal if goal_dist < 1.8 else target
                desired_heading = angle_of(active_target - point)
                heading_error = angle_diff(desired_heading, theta)
                target_v = min(self.v_ref, 0.85 * goal_dist)
                if goal_dist > 0.9:
                    target_v = max(0.2, target_v)
                if abs(heading_error) > 0.7:
                    target_v *= 0.65
                raw = Control(target_v, clamp(1.05 * heading_error, -self.steer_max, self.steer_max))
            control = self._limit_control(raw, prev)
            controls.append(control)

            yaw_rate = control.v / self.wheel_base * math.tan(control.steer)
            theta_mid = theta + 0.5 * yaw_rate * self.dt
            x += control.v * math.cos(theta_mid) * self.dt
            y += control.v * math.sin(theta_mid) * self.dt
            theta += yaw_rate * self.dt
            point = scenario.clamp_point(Point(x, y))
            x = point.x
            y = point.y
            prev = control
        return controls

    def _goal_capture_rollout(self, current: Point, heading: float, scenario: Scenario) -> list[Control]:
        x = current.x
        y = current.y
        theta = heading
        prev = self.prev_control
        controls: list[Control] = []
        for _ in range(self.horizon_steps):
            point = Point(x, y)
            goal_dist = distance(point, scenario.goal)
            if goal_dist <= self.goal_capture_radius:
                raw = Control(0.0, 0.25 * prev.steer)
            else:
                desired_heading = angle_of(scenario.goal - point)
                heading_error = angle_diff(desired_heading, theta)
                target_v = min(self.v_ref, max(0.0, 0.9 * (goal_dist - 0.45 * self.goal_capture_radius)))
                if abs(heading_error) > 0.55:
                    target_v *= 0.55
                raw = Control(target_v, clamp(1.35 * heading_error, -self.steer_max, self.steer_max))
            control = self._limit_control(raw, prev)
            controls.append(control)

            yaw_rate = control.v / self.wheel_base * math.tan(control.steer)
            theta_mid = theta + 0.5 * yaw_rate * self.dt
            x += control.v * math.cos(theta_mid) * self.dt
            y += control.v * math.sin(theta_mid) * self.dt
            theta += yaw_rate * self.dt
            point = scenario.clamp_point(Point(x, y))
            x = point.x
            y = point.y
            prev = control
        return controls

    def _rollout_path(
        self,
        current: Point,
        heading: float,
        start_t: float,
        controls: list[Control],
        scenario: Scenario,
    ) -> list[TimedPoint]:
        x = current.x
        y = current.y
        theta = heading
        path = [TimedPoint(x, y, start_t)]
        for i, control in enumerate(controls, start=1):
            yaw_rate = control.v / self.wheel_base * math.tan(control.steer)
            theta_mid = theta + 0.5 * yaw_rate * self.dt
            x += control.v * math.cos(theta_mid) * self.dt
            y += control.v * math.sin(theta_mid) * self.dt
            theta += yaw_rate * self.dt
            point = scenario.clamp_point(Point(x, y))
            x = point.x
            y = point.y
            path.append(TimedPoint(x, y, start_t + i * self.dt))
        return path

    def _next_heading(self, heading: float, control: Control) -> float:
        yaw_rate = control.v / self.wheel_base * math.tan(control.steer)
        theta = heading + yaw_rate * self.dt
        return math.atan2(math.sin(theta), math.cos(theta))

    def _evaluate_mppi(
        self,
        path: list[TimedPoint],
        controls: list[Control],
        scenario: Scenario,
        prediction_horizon: tuple[PredictionStep, ...],
    ) -> CandidateTrajectory:
        min_clearance = float("inf")
        min_static_clearance = float("inf")
        collision_count = 0
        mean_clearance_penalty = 0.0
        uncertainty_margin_penalty = 0.0
        predicted_collision_penalty = 0.0
        centerline_penalty = 0.0
        heading_recovery_penalty = 0.0
        goal_slowdown_penalty = 0.0
        cp_sum = 0.0
        cp_excess_sq_sum = 0.0
        max_cp = 0.0
        near_term_max_cp = 0.0
        constraint_max_cp = 0.0
        boundary_penalty = 0.0

        for point_index, point in enumerate(path):
            prediction_step = prediction_horizon[point_index]
            true_clearance = self._true_clearance(point.point, scenario, prediction_step)
            min_clearance = min(min_clearance, true_clearance)
            min_static_clearance = min(
                min_static_clearance,
                self._static_clearance(point.point, scenario),
            )
            if true_clearance < 0.0:
                collision_count += 1

            mean_clearance = self._predicted_clearance(point.point, scenario, prediction_step)
            mean_margin = 0.08
            if mean_clearance < mean_margin:
                mean_clearance_penalty += (mean_margin - mean_clearance) ** 2
            if mean_clearance < 0.0:
                predicted_collision_penalty += 1.0 + 4.0 * mean_clearance * mean_clearance

            cp_t = (
                self._joint_collision_probability(point.point, scenario, prediction_step)
                if self.risk_aware
                else 0.0
            )
            cp_sum += cp_t
            max_cp = max(max_cp, cp_t)
            if 0 < point_index <= min(self.horizon_steps, 4):
                near_term_max_cp = max(near_term_max_cp, cp_t)
            if point_index <= self.chance_constraint_steps:
                constraint_max_cp = max(constraint_max_cp, cp_t)
            cp_excess = max(0.0, cp_t - self.cp_threshold)
            cp_excess_sq_sum += cp_excess * cp_excess
            if self.uncertainty_margin_weight > 0.0:
                uncertainty_clearance = self._uncertainty_clearance(
                    point.point,
                    scenario,
                    prediction_step,
                    sigma_scale=self.uncertainty_margin_sigma,
                )
                if uncertainty_clearance < 0.0:
                    uncertainty_margin_penalty += uncertainty_clearance * uncertainty_clearance
            if mean_clearance > 0.75:
                centerline_penalty += self._centerline_deviation(point.point, scenario) ** 2

            if point.x <= scenario.robot_radius or point.x >= scenario.width - scenario.robot_radius:
                boundary_penalty += 5.0
            if point.y <= scenario.robot_radius or point.y >= scenario.height - scenario.robot_radius:
                boundary_penalty += 5.0

        length = polyline_length(path)
        smoothness = path_smoothness(path)
        goal_distances = [distance(point.point, scenario.goal) for point in path]
        final_goal_dist = goal_distances[-1]
        min_goal_dist = min(goal_distances)
        avg_goal_dist = sum(goal_distances) / len(goal_distances)
        closest_goal_index = min(range(len(goal_distances)), key=lambda i: goal_distances[i])
        post_capture_escape_penalty = 0.0
        if min_goal_dist <= self.goal_capture_radius:
            post_capture_escape_penalty = sum(
                max(0.0, goal_dist - self.goal_capture_radius) ** 2
                for goal_dist in goal_distances[closest_goal_index:]
            )
        near_goal_escape_penalty = 0.0
        start_goal_dist = goal_distances[0]
        if start_goal_dist < 2.2:
            allowed_dist = max(start_goal_dist, self.goal_capture_radius) + 0.18
            near_goal_escape_penalty = sum(
                max(0.0, goal_dist - allowed_dist) ** 2
                for goal_dist in goal_distances
            )
        final_progress = self._centerline_progress(path[-1].point, scenario)
        progress_sum = sum(self._centerline_progress(point.point, scenario) for point in path)
        final_heading = math.atan2(
            scenario.goal.y - path[-1].y,
            scenario.goal.x - path[-1].x,
        )
        path_heading = math.atan2(path[-1].y - path[0].y, path[-1].x - path[0].x)
        heading_cost = abs(angle_diff(final_heading, path_heading))
        control_effort = sum(0.05 * (c.v - self.v_ref) ** 2 + 0.45 * c.steer * c.steer for c in controls)
        accel_cost = 0.0
        steer_rate_cost = 0.0
        prev_control = self.prev_control
        for point, control in zip(path[:-1], controls):
            accel_cost += ((control.v - prev_control.v) / self.dt) ** 2
            steer_rate_cost += ((control.steer - prev_control.steer) / self.dt) ** 2
            goal_dist = distance(point.point, scenario.goal)
            if goal_dist < 1.8:
                target_v = clamp(0.75 * goal_dist, 0.0, self.v_ref)
                weight = (1.8 - goal_dist) / 1.8
                goal_slowdown_penalty += weight * (control.v - target_v) ** 2
            prev_control = control
        for point_index, (a, b) in enumerate(zip(path, path[1:]), start=1):
            if self._predicted_clearance(
                b.point,
                scenario,
                prediction_horizon[point_index],
            ) <= 0.75:
                continue
            segment_heading = angle_of(b.point - a.point)
            goal_heading = angle_of(scenario.goal - b.point)
            heading_recovery_penalty += angle_diff(segment_heading, goal_heading) ** 2

        max_cp_excess = max(0.0, max_cp - self.cp_threshold)
        near_term_cp_excess = max(0.0, near_term_max_cp - self.cp_threshold)
        risk_cost = (
            self.lambda_cp * cp_excess_sq_sum
            + self.lambda_hard * max_cp_excess * max_cp_excess
            + 2.5 * self.lambda_hard * near_term_cp_excess * near_term_cp_excess
        )
        recovery_progress = (
            max(0.0, start_goal_dist - final_goal_dist)
            if scenario.recovery_mode
            else 0.0
        )

        cost = (
            14.0 * final_goal_dist
            + 5.0 * min_goal_dist
            + 1.1 * avg_goal_dist
            - 1.5 * final_progress
            - 0.08 * progress_sum
            + 0.35 * length
            + 4.0 * smoothness
            + 2.4 * heading_cost
            + control_effort
            + 0.04 * accel_cost
            + 0.025 * steer_rate_cost
            + 4.0 * goal_slowdown_penalty
            + 18.0 * post_capture_escape_penalty
            + 5.0 * near_goal_escape_penalty
            + 0.06 * centerline_penalty
            + 0.1 * heading_recovery_penalty
            + 65.0 * mean_clearance_penalty
            + self.uncertainty_margin_weight * uncertainty_margin_penalty
            + 350.0 * predicted_collision_penalty
            + boundary_penalty
            + risk_cost
            - 55.0 * recovery_progress
        )
        return CandidateTrajectory(
            path=path,
            cost=cost,
            min_clearance=min_clearance,
            collision_count=collision_count,
            components={
                "goal": final_goal_dist,
                "min_goal_dist": min_goal_dist,
                "final_progress": final_progress,
                "length": length,
                "smoothness": smoothness,
                "mean_clearance_penalty": mean_clearance_penalty,
                "uncertainty_margin_penalty": uncertainty_margin_penalty,
                "predicted_collision_penalty": predicted_collision_penalty,
                "accel_cost": accel_cost,
                "steer_rate_cost": steer_rate_cost,
                "goal_slowdown_penalty": goal_slowdown_penalty,
                "post_capture_escape_penalty": post_capture_escape_penalty,
                "near_goal_escape_penalty": near_goal_escape_penalty,
                "centerline_penalty": centerline_penalty,
                "heading_recovery_penalty": heading_recovery_penalty,
                "recovery_progress": recovery_progress,
                "collision_count": float(collision_count),
                "min_static_clearance": min_static_clearance,
                "cp_sum": cp_sum,
                "cp_excess_sq_sum": cp_excess_sq_sum,
                "max_cp": max_cp,
                "max_cp_excess": max_cp_excess,
                "near_term_max_cp": near_term_max_cp,
                "near_term_cp_excess": near_term_cp_excess,
                "constraint_max_cp": constraint_max_cp,
                "chance_feasible": 1.0 if constraint_max_cp <= self.cp_threshold + 1e-9 else 0.0,
                "risk_cost": risk_cost,
            },
        )

    def _chance_feasible(self, candidate: CandidateTrajectory) -> bool:
        return self._chance_violation(candidate) <= 1e-9

    def _static_feasible(self, candidate: CandidateTrajectory) -> bool:
        return float(candidate.components.get("min_static_clearance", float("inf"))) >= 0.0

    def _chance_violation(self, candidate: CandidateTrajectory) -> float:
        constraint_max_cp = float(candidate.components.get("constraint_max_cp", 0.0))
        return max(0.0, constraint_max_cp - self.cp_threshold)

    def _weights(self, costs: list[float]) -> list[float]:
        best = min(costs)
        raw = [math.exp(-(cost - best) / max(1e-6, self.temperature)) for cost in costs]
        total = sum(raw)
        if total <= 0.0:
            return [1.0 / len(costs) for _ in costs]
        return [value / total for value in raw]

    def _weighted_controls(self, rollouts: list[list[Control]], weights: list[float]) -> list[Control]:
        updated: list[Control] = []
        for i in range(self.horizon_steps):
            v = 0.0
            steer = 0.0
            for rollout, weight in zip(rollouts, weights):
                v += weight * rollout[i].v
                steer += weight * rollout[i].steer
            updated.append(Control(v, steer))
        return self._limit_control_sequence(updated)

    def _limit_control_sequence(self, controls: list[Control]) -> list[Control]:
        limited: list[Control] = []
        prev = self.prev_control
        for control in controls:
            next_control = self._limit_control(control, prev)
            limited.append(next_control)
            prev = next_control
        return limited

    def _limit_control(self, control: Control, prev: Control) -> Control:
        return self._limit_control_with_min(control, prev, self.v_min)

    def _limit_control_with_min(self, control: Control, prev: Control, v_min: float) -> Control:
        dv = clamp(control.v - prev.v, -self.a_max * self.dt, self.a_max * self.dt)
        v = clamp(prev.v + dv, v_min, self.v_max)
        ds = clamp(
            control.steer - prev.steer,
            -self.steer_rate_max * self.dt,
            self.steer_rate_max * self.dt,
        )
        steer = clamp(prev.steer + ds, -self.steer_max, self.steer_max)
        return Control(v, steer)

    def _build_prediction_horizon(
        self,
        start_t: float,
        scenario: Scenario,
    ) -> tuple[PredictionStep, ...]:
        return tuple(
            PredictionStep(
                actual_centers=tuple(obstacle.position_at(t) for obstacle in scenario.dynamic_obstacles),
                predictions=tuple(
                    self._prediction_at_replan_time(obstacle, start_t, t)
                    for obstacle in scenario.dynamic_obstacles
                ),
            )
            for t in (start_t + index * self.dt for index in range(self.horizon_steps + 1))
        )

    def _prediction_at_replan_time(
        self,
        obstacle: DynamicObstacle,
        start_t: float,
        t: float,
    ) -> GaussianPrediction:
        if isinstance(obstacle, DynamicCircleObstacle):
            return obstacle.prediction_at_from(start_t, t)
        return obstacle.prediction_at(t)

    def _true_clearance(
        self,
        point: Point,
        scenario: Scenario,
        prediction_step: PredictionStep,
    ) -> float:
        clearances = [
            distance(point, obstacle.center) - obstacle.radius - scenario.robot_radius
            for obstacle in scenario.static_obstacles
        ]
        clearances.extend(
            distance(point, center) - obstacle.radius - scenario.robot_radius
            for obstacle, center in zip(scenario.dynamic_obstacles, prediction_step.actual_centers)
        )
        return min(clearances) if clearances else float("inf")

    def _static_clearance(
        self,
        point: Point,
        scenario: Scenario,
    ) -> float:
        clearances = [
            distance(point, obstacle.center) - obstacle.radius - scenario.robot_radius
            for obstacle in scenario.static_obstacles
        ]
        return min(clearances) if clearances else float("inf")

    def _predicted_clearance(
        self,
        point: Point,
        scenario: Scenario,
        prediction_step: PredictionStep,
    ) -> float:
        clearances: list[float] = []
        for obstacle in scenario.static_obstacles:
            clearances.append(distance(point, obstacle.center) - obstacle.radius - scenario.robot_radius)
        for obstacle, prediction in zip(scenario.dynamic_obstacles, prediction_step.predictions):
            clearances.append(
                distance(point, prediction.mean)
                - obstacle.radius
                - scenario.robot_radius
                - self.obstacle_inflation_sigma * prediction.max_sigma
            )
        return min(clearances) if clearances else float("inf")

    def _centerline_deviation(self, point: Point, scenario: Scenario) -> float:
        start = scenario.start
        goal = scenario.goal
        line = goal - start
        line_len = norm(line)
        if line_len <= 1e-6:
            return 0.0
        projected = clamp(dot(point - start, line) / (line_len * line_len), 0.0, 1.0)
        closest = Point(start.x + line.x * projected, start.y + line.y * projected)
        return distance(point, closest)

    def _centerline_progress(self, point: Point, scenario: Scenario) -> float:
        start = scenario.start
        goal = scenario.goal
        line = goal - start
        line_len = norm(line)
        if line_len <= 1e-6:
            return 0.0
        projected = clamp(dot(point - start, line) / line_len, 0.0, line_len)
        return projected

    def _goal_seek_tail_control(self, selected_path: list[TimedPoint], scenario: Scenario) -> Control:
        if len(selected_path) < 2:
            return Control(self.v_ref, 0.0)
        tail = selected_path[-1].point
        prev = selected_path[-2].point
        current_heading = angle_of(tail - prev)
        goal_heading = angle_of(scenario.goal - tail)
        error = angle_diff(goal_heading, current_heading)
        goal_dist = distance(tail, scenario.goal)
        v = min(self.v_ref, max(0.0, 0.8 * goal_dist))
        steer = clamp(0.75 * error, -self.steer_max, self.steer_max)
        if abs(error) > 0.9:
            v *= 0.65
        return Control(clamp(v, self.v_min, self.v_max), steer)

    def _uncertainty_clearance(
        self,
        point: Point,
        scenario: Scenario,
        prediction_step: PredictionStep,
        sigma_scale: float,
    ) -> float:
        clearances: list[float] = []
        for obstacle, prediction in zip(scenario.dynamic_obstacles, prediction_step.predictions):
            sigma = prediction.radial_sigma(point, covariance_mode=self.covariance_mode)
            clearances.append(
                distance(point, prediction.mean)
                - obstacle.radius
                - scenario.robot_radius
                - sigma_scale * sigma
            )
        return min(clearances) if clearances else float("inf")

    def _joint_collision_probability(
        self,
        point: Point,
        scenario: Scenario,
        prediction_step: PredictionStep,
    ) -> float:
        no_collision_probability = 1.0
        for obstacle, prediction in zip(scenario.dynamic_obstacles, prediction_step.predictions):
            cp = self._obstacle_collision_probability(
                point,
                prediction,
                obstacle.radius + scenario.robot_radius,
            )
            no_collision_probability *= 1.0 - cp
        return 1.0 - no_collision_probability

    def _obstacle_collision_probability(
        self,
        point: Point,
        prediction: GaussianPrediction,
        threshold: float,
    ) -> float:
        sigma = prediction.radial_sigma(point, covariance_mode=self.covariance_mode)
        if sigma <= 1e-6:
            return 1.0 if distance(point, prediction.mean) < threshold else 0.0
        center_distance = distance(point, prediction.mean)
        z = (threshold - center_distance) / max(1e-6, sigma)
        analytic = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        if self.mc_samples <= 0:
            return clamp(analytic, 0.0, 1.0)

        hits = 0
        for zx, zy in self.normal_samples:
            sample = prediction.sample(
                zx,
                zy,
                covariance_mode=self.covariance_mode,
            )
            if distance(point, sample) < threshold:
                hits += 1
        monte_carlo = hits / len(self.normal_samples)
        return clamp(0.65 * analytic + 0.35 * monte_carlo, 0.0, 1.0)


def make_vanilla_mppi(seed: int = 0, **kwargs: object) -> MPPIPlanner:
    return MPPIPlanner(
        "vanilla_mppi",
        risk_aware=False,
        seed=seed,
        uncertainty_margin_weight=0.0,
        **kwargs,
    )


def make_risk_aware_mppi(seed: int = 0, **kwargs: object) -> MPPIPlanner:
    return MPPIPlanner("risk_aware_mppi", risk_aware=True, seed=seed, **kwargs)


def make_chance_constrained_mppi(seed: int = 0, **kwargs: object) -> MPPIPlanner:
    planner_kwargs = dict(kwargs)
    planner_kwargs["lambda_cp"] = 0.0
    planner_kwargs["lambda_hard"] = 0.0
    planner_kwargs["uncertainty_margin_weight"] = 0.0
    return MPPIPlanner(
        "chance_constrained_mppi",
        risk_aware=True,
        enforce_chance_constraint=True,
        chance_constraint_steps=6,
        seed=seed,
        **planner_kwargs,
    )


def make_uncertainty_margin_mppi(seed: int = 0, **kwargs: object) -> MPPIPlanner:
    return MPPIPlanner(
        "uncertainty_margin_mppi",
        risk_aware=False,
        seed=seed,
        lambda_cp=0.0,
        lambda_hard=0.0,
        **kwargs,
    )


def make_soft_cp_mppi(seed: int = 0, **kwargs: object) -> MPPIPlanner:
    return MPPIPlanner(
        "soft_cp_mppi",
        risk_aware=True,
        seed=seed,
        lambda_hard=0.0,
        uncertainty_margin_weight=0.0,
        **kwargs,
    )


def make_hard_cp_mppi(seed: int = 0, **kwargs: object) -> MPPIPlanner:
    return MPPIPlanner(
        "hard_cp_mppi",
        risk_aware=True,
        seed=seed,
        lambda_cp=0.0,
        uncertainty_margin_weight=0.0,
        **kwargs,
    )


def make_inflated_obstacle_mppi(
    seed: int = 0,
    inflation_sigma: float = 2.0,
    name: str | None = None,
    **kwargs: object,
) -> MPPIPlanner:
    method_name = name or f"inflated_{inflation_sigma:g}sigma_mppi"
    return MPPIPlanner(
        method_name,
        risk_aware=False,
        seed=seed,
        lambda_cp=0.0,
        lambda_hard=0.0,
        uncertainty_margin_weight=0.0,
        obstacle_inflation_sigma=inflation_sigma,
        **kwargs,
    )


def make_isotropic_risk_mppi(seed: int = 0, **kwargs: object) -> MPPIPlanner:
    return MPPIPlanner(
        "isotropic_risk_mppi",
        risk_aware=True,
        seed=seed,
        covariance_mode="isotropic_equivalent",
        **kwargs,
    )


def make_anisotropic_risk_mppi(seed: int = 0, **kwargs: object) -> MPPIPlanner:
    return MPPIPlanner(
        "anisotropic_risk_mppi",
        risk_aware=True,
        seed=seed,
        covariance_mode="model",
        **kwargs,
    )
