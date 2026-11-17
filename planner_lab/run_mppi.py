from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .metrics import Metrics, compute_metrics
from .mppi import make_risk_aware_mppi, make_vanilla_mppi
from .planners import SimulationResult, simulate, simulation_to_dict
from .scenario import mppi_scenarios
from .visualize import render_animation, render_comparison, render_process_strip


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vanilla MPPI vs risk-aware MPPI experiments.")
    parser.add_argument("--out", default="outputs/mppi", help="Output directory.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds per scenario.")
    parser.add_argument("--max-steps", type=int, default=130, help="Maximum closed-loop replanning steps.")
    parser.add_argument("--goal-tolerance", type=float, default=0.6, help="Distance threshold for goal success.")
    parser.add_argument("--uncertainty", type=float, default=1.0, help="Prediction uncertainty scale.")
    parser.add_argument("--rollouts", type=int, default=80, help="MPPI rollouts per replanning step.")
    parser.add_argument("--horizon", type=int, default=18, help="MPPI horizon steps.")
    parser.add_argument("--mc-samples", type=int, default=0, help="Monte Carlo samples for risk-aware CP; 0 uses analytic CP only.")
    parser.add_argument("--lambda-cp", type=float, default=60.0, help="Soft collision-probability cost weight.")
    parser.add_argument("--lambda-hard", type=float, default=180.0, help="Hard cost weight for CP above threshold.")
    parser.add_argument("--cp-threshold", type=float, default=0.1, help="Collision-probability threshold for hard risk cost.")
    parser.add_argument("--wheel-base", type=float, default=0.7, help="Kinematic bicycle wheel base.")
    parser.add_argument("--v-max", type=float, default=1.55, help="Maximum linear speed.")
    parser.add_argument("--a-max", type=float, default=1.2, help="Maximum linear acceleration.")
    parser.add_argument("--steer-max", type=float, default=0.62, help="Maximum steering angle in radians.")
    parser.add_argument("--steer-rate-max", type=float, default=1.6, help="Maximum steering rate in rad/s.")
    parser.add_argument("--min-turn-radius", type=float, default=1.1, help="Minimum turning radius.")
    parser.add_argument(
        "--scenarios",
        default="crossing,corridor,intersection,blocked_line,blocked_crossing",
        help="Comma-separated scenario names to include.",
    )
    parser.add_argument("--save-visual-json", action="store_true", help="Save raw seed-0 visual simulation snapshots.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_names = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    scenario_filter = set(scenario_names)
    if not scenario_filter:
        raise SystemExit("No scenarios selected. Use --scenarios with at least one scenario name.")
    available_scenarios = set(mppi_scenarios(seed=0, uncertainty_scale=args.uncertainty).keys())
    unknown_scenarios = sorted(scenario_filter - available_scenarios)
    if unknown_scenarios:
        available = ", ".join(sorted(available_scenarios))
        unknown = ", ".join(unknown_scenarios)
        raise SystemExit(f"Unknown scenario(s): {unknown}. Available scenarios: {available}")
    write_json(
        out_dir / "run_config.json",
        {
            "runner": "planner_lab.run_mppi",
            "args": vars(args),
            "selected_scenarios": scenario_names,
            "available_scenarios": sorted(available_scenarios),
        },
    )

    trial_rows: list[dict[str, object]] = []
    visual_payload: dict[str, dict[str, object]] = {}

    for seed in range(args.seeds):
        scenarios = mppi_scenarios(seed=seed, uncertainty_scale=args.uncertainty)
        for scenario_name, scenario in scenarios.items():
            if scenario_name not in scenario_filter:
                continue
            common = {
                "horizon_steps": args.horizon,
                "rollouts": args.rollouts,
                "mc_samples": args.mc_samples,
                "lambda_cp": args.lambda_cp,
                "lambda_hard": args.lambda_hard,
                "cp_threshold": args.cp_threshold,
                "wheel_base": args.wheel_base,
                "v_max": args.v_max,
                "a_max": args.a_max,
                "steer_max": args.steer_max,
                "steer_rate_max": args.steer_rate_max,
                "min_turn_radius": args.min_turn_radius,
            }
            planners = [
                make_vanilla_mppi(seed=10000 + seed, **common),
                make_risk_aware_mppi(seed=10000 + seed, **common),
            ]
            results: list[SimulationResult] = []
            metrics: list[Metrics] = []
            for planner in planners:
                result = simulate(
                    planner,
                    scenario,
                    max_steps=args.max_steps,
                    step_dt=planner.dt,
                    goal_tolerance=args.goal_tolerance,
                )
                row = compute_metrics(result, scenario)
                results.append(result)
                metrics.append(row)
                record = row.to_dict()
                record.update(
                    {
                        "scenario": scenario_name,
                        "seed": seed,
                        "uncertainty_scale": args.uncertainty,
                    }
                )
                trial_rows.append(record)
                print(
                    f"{scenario_name} seed={seed} {row.method}: "
                    f"success={row.success}, collision_rate={row.collision_rate:.3f}, "
                    f"min_dist={row.min_safety_distance:.2f}, max_cp={row.max_collision_probability:.2f}, "
                    f"time={row.avg_compute_time_ms:.1f}ms"
                )

            if seed == 0:
                scenario_dir = out_dir / scenario_name
                scenario_dir.mkdir(exist_ok=True)
                render_comparison(
                    scenario_dir / "comparison.svg",
                    scenario,
                    results,
                    metrics,
                    goal_tolerance=args.goal_tolerance,
                )
                render_animation(
                    scenario_dir / "animation.svg",
                    scenario,
                    results,
                    metrics,
                    goal_tolerance=args.goal_tolerance,
                )
                for result in results:
                    render_process_strip(
                        scenario_dir / f"process_{result.method}.svg",
                        scenario,
                        result,
                        panels=5,
                        goal_tolerance=args.goal_tolerance,
                    )
                visual_payload[scenario_name] = {
                    result.method: simulation_to_dict(result)
                    for result in results
                }

    summary_rows = summarize_trials(trial_rows)
    write_csv(out_dir / "trials.csv", trial_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_json(out_dir / "trials.json", trial_rows)
    write_json(out_dir / "summary.json", summary_rows)
    if args.save_visual_json:
        write_json(out_dir / "visual_simulations_seed0.json", visual_payload)

    print("MPPI experiment complete.")
    print(f"Output directory: {out_dir.resolve()}")


def summarize_trials(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["scenario"]), str(row["method"]))].append(row)

    summary: list[dict[str, object]] = []
    for (scenario, method), group in sorted(groups.items()):
        count = len(group)
        summary.append(
            {
                "scenario": scenario,
                "method": method,
                "trials": count,
                "success_rate": mean(1.0 if row["success"] else 0.0 for row in group),
                "reached_at_t": mean_reached_time(group),
                "elapsed_time_s": mean_optional_field(group, "elapsed_time_s"),
                "final_goal_distance": mean_optional_field(group, "final_goal_distance"),
                "min_goal_distance": mean_optional_field(group, "min_goal_distance"),
                "collision_rate": mean(float(row["collision_rate"]) for row in group),
                "min_safety_distance": mean(float(row["min_safety_distance"]) for row in group),
                "average_speed": mean(float(row["average_speed"]) for row in group),
                "trajectory_length": mean(float(row["trajectory_length"]) for row in group),
                "max_collision_probability": mean(float(row["max_collision_probability"]) for row in group),
                "avg_compute_time_ms": mean(float(row["avg_compute_time_ms"]) for row in group),
                "max_compute_time_ms": mean(float(row["max_compute_time_ms"]) for row in group),
            }
        )
    return summary


def mean(values: object) -> float:
    rows = list(values)
    return sum(rows) / max(1, len(rows))


def mean_reached_time(rows: list[dict[str, object]]) -> float | None:
    values = [float(row["reached_at_t"]) for row in rows if row["reached_at_t"] is not None]
    if not values:
        return None
    return sum(values) / len(values)


def mean_optional_field(rows: list[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
