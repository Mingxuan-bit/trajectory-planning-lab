from __future__ import annotations

import unittest
from unittest.mock import patch

from planner_lab.geometry import Point, TimedPoint
from planner_lab.mppi import Control, MPPIPlanner
from planner_lab.planners import CandidateTrajectory
from planner_lab.scenario import CircleObstacle, Scenario


class MPPIStaticSafetyTest(unittest.TestCase):
    def test_plan_rejects_lower_cost_path_that_intersects_static_obstacle(self) -> None:
        scenario = Scenario(
            width=6.0,
            height=5.0,
            start=Point(1.0, 2.0),
            goal=Point(5.0, 2.0),
            robot_radius=0.25,
            static_obstacles=(CircleObstacle("service_island", Point(2.0, 2.0), 0.5),),
            dynamic_obstacles=(),
        )
        planner = MPPIPlanner(
            "test",
            risk_aware=False,
            seed=0,
            horizon_steps=2,
            rollouts=2,
            mc_samples=0,
        )
        controls = [Control(0.8, 0.0), Control(0.8, 0.0)]
        blocked_path = [
            TimedPoint(1.0, 2.0, 0.0),
            TimedPoint(2.0, 2.0, planner.dt),
            TimedPoint(3.0, 2.0, 2.0 * planner.dt),
        ]
        detour_path = [
            TimedPoint(1.0, 2.0, 0.0),
            TimedPoint(1.6, 3.0, planner.dt),
            TimedPoint(2.7, 3.0, 2.0 * planner.dt),
        ]

        with (
            patch.object(planner, "_sample_rollouts", return_value=[controls, controls]),
            patch.object(
                planner,
                "_rollout_path",
                side_effect=[blocked_path, detour_path, blocked_path],
            ),
        ):
            result = planner.plan(scenario.start, 0.0, 0.0, scenario)

        self.assertLess(
            float(planner._evaluate_mppi(
                blocked_path,
                controls,
                scenario,
                planner._build_prediction_horizon(0.0, scenario),
            ).components["min_static_clearance"]),
            0.0,
        )
        self.assertGreaterEqual(
            float(result.selected.components["min_static_clearance"]),
            0.0,
        )
        self.assertEqual(result.selected.path, detour_path)

    def test_recovery_rollouts_include_reverse_then_forward_escape(self) -> None:
        planner = MPPIPlanner(
            "test",
            risk_aware=False,
            seed=0,
            horizon_steps=9,
            rollouts=20,
            mc_samples=0,
            v_ref=1.0,
        )

        recovery_rollouts = planner._recovery_rollouts()

        self.assertTrue(
            any(
                any(control.v < -1e-6 for control in controls[:4])
                and any(control.v > 1e-6 for control in controls[4:])
                for controls in recovery_rollouts
            )
        )


class MPPIChanceSafetyFallbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = Scenario(
            width=6.0,
            height=5.0,
            start=Point(1.0, 2.0),
            goal=Point(5.0, 2.0),
            robot_radius=0.25,
            static_obstacles=(),
            dynamic_obstacles=(),
        )
        self.controls = [Control(0.8, 0.0), Control(0.8, 0.0)]
        self.forward_path = [
            TimedPoint(1.0, 2.0, 0.0),
            TimedPoint(1.3, 2.0, 0.18),
            TimedPoint(1.6, 2.0, 0.36),
        ]
        self.brake_path = [
            TimedPoint(1.0, 2.0, 0.0),
            TimedPoint(1.1, 2.0, 0.18),
            TimedPoint(1.15, 2.0, 0.36),
        ]

    def _candidate(
        self,
        path: list[TimedPoint],
        cost: float,
        static_clearance: float = 1.0,
    ) -> CandidateTrajectory:
        return CandidateTrajectory(
            path=path,
            cost=cost,
            min_clearance=0.2,
            collision_count=0,
            components={
                "constraint_max_cp": 0.25,
                "min_static_clearance": static_clearance,
            },
        )

    def _plan_with_all_candidates_infeasible(
        self,
        fallback: str,
        brake_static_clearance: float = 1.0,
    ) -> CandidateTrajectory:
        planner = MPPIPlanner(
            "test",
            risk_aware=True,
            seed=0,
            horizon_steps=2,
            rollouts=1,
            mc_samples=0,
            cp_threshold=0.10,
            enforce_chance_constraint=True,
            chance_safety_fallback=fallback,
        )
        with (
            patch.object(planner, "_sample_rollouts", return_value=[self.controls]),
            patch.object(
                planner,
                "_rollout_path",
                side_effect=[self.forward_path, self.forward_path, self.brake_path],
            ),
            patch.object(
                planner,
                "_evaluate_mppi",
                side_effect=[
                    self._candidate(self.forward_path, 1.0),
                    self._candidate(self.forward_path, 2.0),
                    self._candidate(
                        self.brake_path,
                        3.0,
                        static_clearance=brake_static_clearance,
                    ),
                ],
            ),
        ):
            return planner.plan(self.scenario.start, 0.0, 0.0, self.scenario).selected

    def test_default_least_violation_preserves_previous_selection(self) -> None:
        selected = self._plan_with_all_candidates_infeasible("least_violation")

        self.assertEqual(selected.path, self.forward_path)
        self.assertEqual(selected.components["chance_safety_fallback_used"], 0.0)

    def test_brake_fallback_selects_static_safe_braking_rollout(self) -> None:
        selected = self._plan_with_all_candidates_infeasible("brake")

        self.assertEqual(selected.path, self.brake_path)
        self.assertEqual(selected.components["chance_safety_fallback_used"], 1.0)

    def test_brake_fallback_preserves_static_safety_screen(self) -> None:
        selected = self._plan_with_all_candidates_infeasible(
            "brake",
            brake_static_clearance=-0.1,
        )

        self.assertEqual(selected.path, self.forward_path)
        self.assertEqual(selected.components["chance_safety_fallback_used"], 0.0)

    def test_unknown_fallback_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MPPIPlanner(
                "test",
                risk_aware=True,
                chance_safety_fallback="not_a_policy",
            )


if __name__ == "__main__":
    unittest.main()
