from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .metrics import compute_metrics
from .planners import ElasticBandPlanner, SamplingPlanner, simulate, simulation_to_dict
from .scenario import default_scenario
from .visualize import render_comparison, render_process_strip


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 2D trajectory-planning experiment.")
    parser.add_argument("--out", default="outputs/basic", help="Output directory.")
    parser.add_argument("--max-steps", type=int, default=150, help="Maximum replanning steps.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = default_scenario()
    planners = [SamplingPlanner(), ElasticBandPlanner()]
    results = [simulate(planner, scenario, max_steps=args.max_steps) for planner in planners]
    metrics = [compute_metrics(result, scenario) for result in results]

    metrics_rows = [row.to_dict() for row in metrics]
    (out_dir / "metrics.json").write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(metrics_rows)

    simulations = {
        result.method: simulation_to_dict(result)
        for result in results
    }
    (out_dir / "simulations.json").write_text(json.dumps(simulations, indent=2), encoding="utf-8")

    render_comparison(out_dir / "comparison.svg", scenario, results, metrics, goal_tolerance=0.42)
    for result in results:
        render_process_strip(out_dir / f"process_{result.method}.svg", scenario, result, goal_tolerance=0.42)

    print("Experiment complete.")
    print(f"Output directory: {out_dir.resolve()}")
    for row in metrics:
        reached = f"{row.reached_at_t:.1f}s" if row.reached_at_t is not None else "not reached"
        print(
            f"- {row.method}: success={row.success}, reached={reached}, "
            f"length={row.trajectory_length:.2f}, min_clearance={row.min_safety_distance:.2f}, "
            f"collision_rate={row.collision_rate:.3f}, avg_time={row.avg_compute_time_ms:.2f}ms"
        )


if __name__ == "__main__":
    main()
