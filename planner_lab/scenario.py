from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .geometry import EPS, Point, TimedPoint, distance, dot, norm, rotate


PREDICTION_MODELS = ("isotropic", "velocity_aligned")


@dataclass(frozen=True)
class GaussianPrediction:
    mean: Point
    longitudinal: Point
    lateral: Point
    sigma_longitudinal: float
    sigma_lateral: float
    covariance_xx: float
    covariance_xy: float
    covariance_yy: float

    @property
    def equivalent_sigma(self) -> float:
        return math.sqrt(self.sigma_longitudinal * self.sigma_lateral)

    @property
    def max_sigma(self) -> float:
        return max(self.sigma_longitudinal, self.sigma_lateral)

    @property
    def angle(self) -> float:
        return math.atan2(self.longitudinal.y, self.longitudinal.x)

    def radial_sigma(self, point: Point, covariance_mode: str = "model") -> float:
        if covariance_mode == "isotropic_equivalent":
            return self.equivalent_sigma
        if covariance_mode == "isotropic_max":
            return self.max_sigma
        if covariance_mode != "model":
            raise ValueError(f"Unknown covariance mode: {covariance_mode}")
        radial = point - self.mean
        radial_norm = norm(radial)
        if radial_norm <= EPS:
            return self.max_sigma
        direction = radial / radial_norm
        return math.sqrt(
            (dot(direction, self.longitudinal) * self.sigma_longitudinal) ** 2
            + (dot(direction, self.lateral) * self.sigma_lateral) ** 2
        )

    def sample(
        self,
        z_longitudinal: float,
        z_lateral: float,
        covariance_mode: str = "model",
    ) -> Point:
        if covariance_mode == "isotropic_equivalent":
            sigma = self.equivalent_sigma
            return Point(self.mean.x + sigma * z_longitudinal, self.mean.y + sigma * z_lateral)
        if covariance_mode == "isotropic_max":
            sigma = self.max_sigma
            return Point(self.mean.x + sigma * z_longitudinal, self.mean.y + sigma * z_lateral)
        if covariance_mode != "model":
            raise ValueError(f"Unknown covariance mode: {covariance_mode}")
        if abs(self.sigma_longitudinal - self.sigma_lateral) <= EPS:
            return Point(
                self.mean.x + self.sigma_longitudinal * z_longitudinal,
                self.mean.y + self.sigma_lateral * z_lateral,
            )
        return (
            self.mean
            + self.longitudinal * (self.sigma_longitudinal * z_longitudinal)
            + self.lateral * (self.sigma_lateral * z_lateral)
        )


@dataclass(frozen=True)
class CircleObstacle:
    name: str
    center: Point
    radius: float

    def position_at(self, _t: float) -> Point:
        return self.center

    @property
    def is_dynamic(self) -> bool:
        return False


@dataclass(frozen=True)
class DynamicCircleObstacle:
    name: str
    start: Point
    velocity: Point
    radius: float
    t0: float = 0.0
    actual_velocity: Point | None = None
    prediction_sigma0: float = 0.0
    prediction_sigma_v: float = 0.0
    prediction_longitudinal_scale: float = 1.0
    prediction_lateral_scale: float = 1.0

    def position_at(self, t: float) -> Point:
        dt = max(0.0, t - self.t0)
        velocity = self.actual_velocity or self.velocity
        return Point(self.start.x + velocity.x * dt, self.start.y + velocity.y * dt)

    def predicted_position_at(self, t: float) -> Point:
        dt = max(0.0, t - self.t0)
        return Point(self.start.x + self.velocity.x * dt, self.start.y + self.velocity.y * dt)

    def predicted_position_from(self, observed_t: float, t: float) -> Point:
        if observed_t < self.t0:
            if t <= self.t0:
                return self.start
            return Point(
                self.start.x + self.velocity.x * (t - self.t0),
                self.start.y + self.velocity.y * (t - self.t0),
            )
        observed = self.position_at(observed_t)
        dt = max(0.0, t - observed_t)
        return Point(observed.x + self.velocity.x * dt, observed.y + self.velocity.y * dt)

    def prediction_sigma_at(self, t: float) -> float:
        dt = max(0.0, t - self.t0)
        return self.prediction_sigma0 + self.prediction_sigma_v * dt

    def prediction_sigma_from(self, observed_t: float, t: float) -> float:
        dt = max(0.0, t - observed_t)
        return self.prediction_sigma0 + self.prediction_sigma_v * dt

    def prediction_at(self, t: float) -> GaussianPrediction:
        mean = self.predicted_position_at(t)
        return self._prediction(mean, self.prediction_sigma_at(t))

    def prediction_at_from(self, observed_t: float, t: float) -> GaussianPrediction:
        mean = self.predicted_position_from(observed_t, t)
        return self._prediction(mean, self.prediction_sigma_from(observed_t, t))

    def _prediction(self, mean: Point, sigma: float) -> GaussianPrediction:
        longitudinal, lateral = self.prediction_axes()
        sigma_longitudinal = sigma * self.prediction_longitudinal_scale
        sigma_lateral = sigma * self.prediction_lateral_scale
        longitudinal_variance = sigma_longitudinal**2
        lateral_variance = sigma_lateral**2
        covariance_xx = (
            longitudinal.x**2 * longitudinal_variance
            + lateral.x**2 * lateral_variance
        )
        covariance_xy = (
            longitudinal.x * longitudinal.y * longitudinal_variance
            + lateral.x * lateral.y * lateral_variance
        )
        covariance_yy = (
            longitudinal.y**2 * longitudinal_variance
            + lateral.y**2 * lateral_variance
        )
        return GaussianPrediction(
            mean=mean,
            longitudinal=longitudinal,
            lateral=lateral,
            sigma_longitudinal=sigma_longitudinal,
            sigma_lateral=sigma_lateral,
            covariance_xx=covariance_xx,
            covariance_xy=covariance_xy,
            covariance_yy=covariance_yy,
        )

    def prediction_axes(self) -> tuple[Point, Point]:
        speed = norm(self.velocity)
        longitudinal = self.velocity / speed if speed > EPS else Point(1.0, 0.0)
        return longitudinal, rotate(longitudinal, math.pi / 2.0)

    def prediction_sigmas_at(self, t: float) -> tuple[float, float]:
        sigma = self.prediction_sigma_at(t)
        return sigma * self.prediction_longitudinal_scale, sigma * self.prediction_lateral_scale

    def prediction_equivalent_sigma_at(self, t: float) -> float:
        return self.prediction_at(t).equivalent_sigma

    def prediction_max_sigma_at(self, t: float) -> float:
        return self.prediction_at(t).max_sigma

    def prediction_radial_sigma_at(self, point: Point, t: float, covariance_mode: str = "model") -> float:
        return self.prediction_at(t).radial_sigma(point, covariance_mode)

    def prediction_sample_at(
        self,
        t: float,
        z_longitudinal: float,
        z_lateral: float,
        covariance_mode: str = "model",
    ) -> Point:
        return self.prediction_at(t).sample(z_longitudinal, z_lateral, covariance_mode)

    @property
    def is_dynamic(self) -> bool:
        return True


@dataclass(frozen=True)
class TrajectoryDynamicObstacle:
    name: str
    trajectory: tuple[TimedPoint, ...]
    radius: float
    prediction_sigma0: float = 0.0
    prediction_sigma_v: float = 0.0
    prediction_longitudinal_scale: float = 1.0
    prediction_lateral_scale: float = 1.0

    def position_at(self, t: float) -> Point:
        return self._interpolate(t)

    def predicted_position_at(self, t: float) -> Point:
        return self._interpolate(t)

    def prediction_sigma_at(self, t: float) -> float:
        t0 = self.trajectory[0].t if self.trajectory else 0.0
        dt = max(0.0, t - t0)
        return self.prediction_sigma0 + self.prediction_sigma_v * dt

    def prediction_at(self, t: float) -> GaussianPrediction:
        mean = self.predicted_position_at(t)
        longitudinal, lateral = self.prediction_axes_at(t)
        sigma_longitudinal, sigma_lateral = self.prediction_sigmas_at(t)
        longitudinal_variance = sigma_longitudinal**2
        lateral_variance = sigma_lateral**2
        covariance_xx = (
            longitudinal.x**2 * longitudinal_variance
            + lateral.x**2 * lateral_variance
        )
        covariance_xy = (
            longitudinal.x * longitudinal.y * longitudinal_variance
            + lateral.x * lateral.y * lateral_variance
        )
        covariance_yy = (
            longitudinal.y**2 * longitudinal_variance
            + lateral.y**2 * lateral_variance
        )
        return GaussianPrediction(
            mean=mean,
            longitudinal=longitudinal,
            lateral=lateral,
            sigma_longitudinal=sigma_longitudinal,
            sigma_lateral=sigma_lateral,
            covariance_xx=covariance_xx,
            covariance_xy=covariance_xy,
            covariance_yy=covariance_yy,
        )

    def prediction_axes_at(self, t: float) -> tuple[Point, Point]:
        velocity = self._velocity_at(t)
        speed = norm(velocity)
        longitudinal = velocity / speed if speed > EPS else Point(1.0, 0.0)
        return longitudinal, rotate(longitudinal, math.pi / 2.0)

    def prediction_sigmas_at(self, t: float) -> tuple[float, float]:
        sigma = self.prediction_sigma_at(t)
        return sigma * self.prediction_longitudinal_scale, sigma * self.prediction_lateral_scale

    def prediction_equivalent_sigma_at(self, t: float) -> float:
        return self.prediction_at(t).equivalent_sigma

    def prediction_max_sigma_at(self, t: float) -> float:
        return self.prediction_at(t).max_sigma

    def prediction_radial_sigma_at(self, point: Point, t: float, covariance_mode: str = "model") -> float:
        return self.prediction_at(t).radial_sigma(point, covariance_mode)

    def prediction_sample_at(
        self,
        t: float,
        z_longitudinal: float,
        z_lateral: float,
        covariance_mode: str = "model",
    ) -> Point:
        return self.prediction_at(t).sample(z_longitudinal, z_lateral, covariance_mode)

    @property
    def is_dynamic(self) -> bool:
        return True

    def _interpolate(self, t: float) -> Point:
        if not self.trajectory:
            return Point(0.0, 0.0)
        if t <= self.trajectory[0].t:
            return self.trajectory[0].point
        if t >= self.trajectory[-1].t:
            return self.trajectory[-1].point
        for before, after in zip(self.trajectory, self.trajectory[1:]):
            if before.t <= t <= after.t:
                dt = after.t - before.t
                if dt <= EPS:
                    return after.point
                u = (t - before.t) / dt
                return Point(
                    before.x + (after.x - before.x) * u,
                    before.y + (after.y - before.y) * u,
                )
        return self.trajectory[-1].point

    def _velocity_at(self, t: float) -> Point:
        if len(self.trajectory) < 2:
            return Point(0.0, 0.0)
        if t <= self.trajectory[0].t:
            before, after = self.trajectory[0], self.trajectory[1]
        elif t >= self.trajectory[-1].t:
            before, after = self.trajectory[-2], self.trajectory[-1]
        else:
            before, after = self.trajectory[-2], self.trajectory[-1]
            for left, right in zip(self.trajectory, self.trajectory[1:]):
                if left.t <= t <= right.t:
                    before, after = left, right
                    break
        dt = after.t - before.t
        if dt <= EPS:
            return Point(0.0, 0.0)
        return Point((after.x - before.x) / dt, (after.y - before.y) / dt)


DynamicObstacle = DynamicCircleObstacle | TrajectoryDynamicObstacle
Obstacle = CircleObstacle | DynamicObstacle


@dataclass(frozen=True)
class Scenario:
    width: float
    height: float
    start: Point
    goal: Point
    robot_radius: float
    static_obstacles: tuple[CircleObstacle, ...]
    dynamic_obstacles: tuple[DynamicObstacle, ...]
    recovery_mode: bool = False

    @property
    def obstacles(self) -> tuple[Obstacle, ...]:
        return (*self.static_obstacles, *self.dynamic_obstacles)

    def clearance(self, point: Point, t: float) -> float:
        if not self.obstacles:
            return float("inf")
        return min(
            distance(point, obstacle.position_at(t)) - obstacle.radius - self.robot_radius
            for obstacle in self.obstacles
        )

    def collides(self, point: Point, t: float) -> bool:
        return self.clearance(point, t) < 0.0

    def clamp_point(self, point: Point) -> Point:
        margin = self.robot_radius
        return Point(
            min(self.width - margin, max(margin, point.x)),
            min(self.height - margin, max(margin, point.y)),
        )


def default_scenario() -> Scenario:
    return Scenario(
        width=18.0,
        height=10.0,
        start=Point(1.0, 1.0),
        goal=Point(16.7, 8.5),
        robot_radius=0.28,
        static_obstacles=(
            CircleObstacle("pillar_a", Point(5.1, 4.5), 1.1),
            CircleObstacle("pillar_b", Point(9.4, 6.8), 1.25),
            CircleObstacle("pillar_c", Point(12.4, 3.6), 1.05),
        ),
        dynamic_obstacles=(
            DynamicCircleObstacle("crossing_1", Point(7.4, 0.8), Point(0.0, 0.48), 0.48),
            DynamicCircleObstacle("crossing_2", Point(14.8, 9.2), Point(-0.38, -0.34), 0.55),
        ),
    )


def mppi_crossing_scenario(
    seed: int = 0,
    uncertainty_scale: float = 1.0,
    prediction_model: str = "isotropic",
) -> Scenario:
    rng = random.Random(seed)

    def jittered_velocity(vx: float, vy: float, scale: float = 0.04) -> Point:
        return perturb_velocity(rng, Point(vx, vy), scale, prediction_model)

    longitudinal_scale, lateral_scale = prediction_scales(prediction_model)

    return Scenario(
        width=14.0,
        height=8.0,
        start=Point(1.0, 4.0),
        goal=Point(13.0, 4.0),
        robot_radius=0.28,
        static_obstacles=(),
        dynamic_obstacles=(
            DynamicCircleObstacle(
                "ped_cross",
                Point(6.7, 0.6 + rng.uniform(-0.25, 0.25)),
                Point(0.0, 0.63),
                0.34,
                actual_velocity=jittered_velocity(0.0, 0.63),
                prediction_sigma0=0.14 * uncertainty_scale,
                prediction_sigma_v=0.09 * uncertainty_scale,
                prediction_longitudinal_scale=longitudinal_scale,
                prediction_lateral_scale=lateral_scale,
            ),
            DynamicCircleObstacle(
                "bike_diag",
                Point(11.5, 7.2 + rng.uniform(-0.2, 0.2)),
                Point(-0.42, -0.31),
                0.38,
                actual_velocity=jittered_velocity(-0.42, -0.31),
                prediction_sigma0=0.12 * uncertainty_scale,
                prediction_sigma_v=0.07 * uncertainty_scale,
                prediction_longitudinal_scale=longitudinal_scale,
                prediction_lateral_scale=lateral_scale,
            ),
        ),
    )


def mppi_corridor_scenario(
    seed: int = 0,
    uncertainty_scale: float = 1.0,
    prediction_model: str = "isotropic",
) -> Scenario:
    rng = random.Random(seed)
    longitudinal_scale, lateral_scale = prediction_scales(prediction_model)
    obstacles: list[DynamicCircleObstacle] = []
    base_rows = [
        ("ped_a", Point(4.2, 6.8), Point(0.28, -0.35), 0.32),
        ("ped_b", Point(7.4, 1.0), Point(-0.12, 0.48), 0.34),
        ("ped_c", Point(10.6, 6.9), Point(-0.35, -0.28), 0.33),
        ("ped_d", Point(12.6, 2.0), Point(-0.42, 0.18), 0.36),
    ]
    for name, start, velocity, radius in base_rows:
        obstacles.append(
            DynamicCircleObstacle(
                name,
                Point(start.x + rng.uniform(-0.3, 0.3), start.y + rng.uniform(-0.25, 0.25)),
                velocity,
                radius,
                actual_velocity=perturb_velocity(rng, velocity, 0.04, prediction_model),
                prediction_sigma0=0.13 * uncertainty_scale,
                prediction_sigma_v=0.08 * uncertainty_scale,
                prediction_longitudinal_scale=longitudinal_scale,
                prediction_lateral_scale=lateral_scale,
            )
        )

    return Scenario(
        width=15.0,
        height=8.0,
        start=Point(1.0, 1.2),
        goal=Point(14.0, 6.8),
        robot_radius=0.28,
        static_obstacles=(
            CircleObstacle("wall_low", Point(7.5, 0.25), 0.18),
            CircleObstacle("wall_high", Point(7.5, 7.75), 0.18),
        ),
        dynamic_obstacles=tuple(obstacles),
    )


def mppi_intersection_scenario(
    seed: int = 0,
    uncertainty_scale: float = 1.0,
    prediction_model: str = "isotropic",
) -> Scenario:
    rng = random.Random(seed)
    longitudinal_scale, lateral_scale = prediction_scales(prediction_model)
    rows = [
        ("veh_left", Point(0.8, 3.6), Point(0.62, 0.0), 0.42),
        ("veh_top", Point(8.0, 7.4), Point(0.0, -0.48), 0.42),
        ("ped_right", Point(13.5, 5.6), Point(-0.46, -0.09), 0.34),
    ]
    obstacles = []
    for name, start, velocity, radius in rows:
        obstacles.append(
            DynamicCircleObstacle(
                name,
                Point(start.x + rng.uniform(-0.2, 0.2), start.y + rng.uniform(-0.2, 0.2)),
                velocity,
                radius,
                actual_velocity=perturb_velocity(rng, velocity, 0.035, prediction_model),
                prediction_sigma0=0.12 * uncertainty_scale,
                prediction_sigma_v=0.1 * uncertainty_scale,
                prediction_longitudinal_scale=longitudinal_scale,
                prediction_lateral_scale=lateral_scale,
            )
        )

    return Scenario(
        width=15.0,
        height=8.0,
        start=Point(1.0, 1.0),
        goal=Point(13.8, 7.0),
        robot_radius=0.28,
        static_obstacles=(
            CircleObstacle("island_a", Point(5.2, 5.9), 0.55),
            CircleObstacle("island_b", Point(9.8, 2.2), 0.55),
        ),
        dynamic_obstacles=tuple(obstacles),
    )


def mppi_static_block_scenario(
    seed: int = 0,
    uncertainty_scale: float = 1.0,
    prediction_model: str = "isotropic",
) -> Scenario:
    rng = random.Random(seed)
    prediction_scales(prediction_model)
    return Scenario(
        width=14.0,
        height=8.0,
        start=Point(1.0, 4.0),
        goal=Point(13.0, 4.0),
        robot_radius=0.28,
        static_obstacles=(
            CircleObstacle("central_block", Point(7.0, 4.0 + rng.uniform(-0.15, 0.15)), 0.95),
        ),
        dynamic_obstacles=(),
    )


def mppi_blocked_crossing_scenario(
    seed: int = 0,
    uncertainty_scale: float = 1.0,
    prediction_model: str = "isotropic",
) -> Scenario:
    rng = random.Random(seed)
    longitudinal_scale, lateral_scale = prediction_scales(prediction_model)
    return Scenario(
        width=14.0,
        height=8.0,
        start=Point(1.0, 4.0),
        goal=Point(13.0, 4.0),
        robot_radius=0.28,
        static_obstacles=(
            CircleObstacle("central_block", Point(6.7, 4.0 + rng.uniform(-0.12, 0.12)), 0.82),
            CircleObstacle("upper_slow_zone", Point(8.9, 5.8), 0.48),
        ),
        dynamic_obstacles=(
            DynamicCircleObstacle(
                "ped_gap",
                Point(8.0 + rng.uniform(-0.15, 0.15), 0.9),
                Point(0.0, 0.48),
                0.34,
                actual_velocity=perturb_velocity(
                    rng,
                    Point(0.0, 0.48),
                    0.035,
                    prediction_model,
                    secondary_scale=0.04,
                ),
                prediction_sigma0=0.14 * uncertainty_scale,
                prediction_sigma_v=0.09 * uncertainty_scale,
                prediction_longitudinal_scale=longitudinal_scale,
                prediction_lateral_scale=lateral_scale,
            ),
            DynamicCircleObstacle(
                "bike_diag",
                Point(11.8, 6.9 + rng.uniform(-0.2, 0.2)),
                Point(-0.38, -0.26),
                0.36,
                actual_velocity=perturb_velocity(
                    rng,
                    Point(-0.38, -0.26),
                    0.04,
                    prediction_model,
                    secondary_scale=0.035,
                ),
                prediction_sigma0=0.12 * uncertainty_scale,
                prediction_sigma_v=0.08 * uncertainty_scale,
                prediction_longitudinal_scale=longitudinal_scale,
                prediction_lateral_scale=lateral_scale,
            ),
        ),
    )


def mppi_scenarios(
    seed: int = 0,
    uncertainty_scale: float = 1.0,
    prediction_model: str = "isotropic",
) -> dict[str, Scenario]:
    return {
        "crossing": mppi_crossing_scenario(seed, uncertainty_scale, prediction_model),
        "corridor": mppi_corridor_scenario(seed, uncertainty_scale, prediction_model),
        "intersection": mppi_intersection_scenario(seed, uncertainty_scale, prediction_model),
        "blocked_line": mppi_static_block_scenario(seed, uncertainty_scale, prediction_model),
        "blocked_crossing": mppi_blocked_crossing_scenario(seed, uncertainty_scale, prediction_model),
    }


def prediction_scales(prediction_model: str) -> tuple[float, float]:
    if prediction_model == "isotropic":
        return 1.0, 1.0
    if prediction_model == "velocity_aligned":
        return 1.6, 0.6
    raise ValueError(f"Unknown prediction model: {prediction_model}. Available: {', '.join(PREDICTION_MODELS)}")


def perturb_velocity(
    rng: random.Random,
    velocity: Point,
    base_scale: float,
    prediction_model: str,
    secondary_scale: float | None = None,
) -> Point:
    longitudinal_scale, lateral_scale = prediction_scales(prediction_model)
    second_scale = secondary_scale if secondary_scale is not None else base_scale
    if prediction_model == "isotropic":
        return Point(
            velocity.x + rng.gauss(0.0, base_scale),
            velocity.y + rng.gauss(0.0, second_scale),
        )
    aligned_base_scale = 0.5 * (base_scale + second_scale)
    speed = norm(velocity)
    longitudinal = velocity / speed if speed > EPS else Point(1.0, 0.0)
    lateral = rotate(longitudinal, math.pi / 2.0)
    return (
        velocity
        + longitudinal * rng.gauss(0.0, aligned_base_scale * longitudinal_scale)
        + lateral * rng.gauss(0.0, aligned_base_scale * lateral_scale)
    )


def obstacle_snapshot(scenario: Scenario, t: float) -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    for obstacle in scenario.obstacles:
        p = obstacle.position_at(t)
        rows.append(
            {
                "name": obstacle.name,
                "x": p.x,
                "y": p.y,
                "r": obstacle.radius,
                "dynamic": obstacle.is_dynamic,
                **(
                    {
                        "sigma_longitudinal": obstacle.prediction_sigmas_at(t)[0],
                        "sigma_lateral": obstacle.prediction_sigmas_at(t)[1],
                    }
                        if isinstance(obstacle, (DynamicCircleObstacle, TrajectoryDynamicObstacle))
                    else {}
                ),
            }
        )
    return rows


def timed_point_to_dict(point: TimedPoint) -> dict[str, float]:
    return {"x": point.x, "y": point.y, "t": point.t}
