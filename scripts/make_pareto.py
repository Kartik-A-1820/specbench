"""
scripts/make_pareto.py — produce the Pareto curve from benchmark results.

Reads results/raw_measurements.jsonl, aggregates per configuration,
and produces:
  - results/pareto.csv: one row per configuration with mean_acceptance_length,
    speedup, and CI bounds.
  - results/figures/pareto_acceptance_vs_speedup.png: the headline plot.

The plot shows mean acceptance length on the x-axis and wall-clock speedup
on the y-axis, one point per configuration, with error bars from bootstrap CIs.
The Pareto frontier is marked. The Phase 1 vanilla speculative decoding point
is included for comparison.

Usage:
    python scripts/make_pareto.py
    python scripts/make_pareto.py --results results/raw_measurements.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
RAW_JSONL = RESULTS_DIR / "raw_measurements.jsonl"
PARETO_CSV = RESULTS_DIR / "pareto.csv"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_records(path: Path) -> list:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def aggregate_config(records: list) -> dict:
    """
    Aggregate records for a single configuration (mode + K + draft_model).

    Returns dict with speedup estimate, CI, and acceptance length.
    """
    from specbench.stats import bootstrap_speedup_ci, bootstrap_mean_ci

    # Separate baseline and spec records by prompt_id.
    baseline = {r["prompt_id"]: r for r in records if r["mode"] == "baseline"}
    spec = {r["prompt_id"]: r for r in records if r["mode"] != "baseline"}

    if not baseline or not spec:
        return {}

    common_prompts = sorted(set(baseline.keys()) & set(spec.keys()))
    if len(common_prompts) < 5:
        return {}

    baseline_tps = np.array([baseline[p]["tokens_per_s"] for p in common_prompts])
    spec_tps = np.array([spec[p]["tokens_per_s"] for p in common_prompts])
    acceptance_lengths = np.array([spec[p]["mean_acceptance_length"] for p in common_prompts])

    rng = np.random.default_rng(42)
    speedup, ci_lo, ci_hi = bootstrap_speedup_ci(baseline_tps, spec_tps, rng=rng)
    alpha_mean, alpha_ci_lo, alpha_ci_hi = bootstrap_mean_ci(acceptance_lengths, rng=rng)

    spec_record = next(iter(spec.values()))
    return {
        "mode": spec_record["mode"],
        "draft_model": spec_record.get("draft_model", ""),
        "K": spec_record.get("K", 0),
        "mean_acceptance_length": alpha_mean,
        "acceptance_length_ci_lo": alpha_ci_lo,
        "acceptance_length_ci_hi": alpha_ci_hi,
        "speedup": speedup,
        "speedup_ci_lo": ci_lo,
        "speedup_ci_hi": ci_hi,
        "n_prompts": len(common_prompts),
        "baseline_mean_tps": float(baseline_tps.mean()),
        "spec_mean_tps": float(spec_tps.mean()),
    }


def pareto_frontier(points: list) -> list:
    """
    Return the Pareto-optimal points (maximising both x and y).
    A point is Pareto-optimal if no other point dominates it on both axes.
    """
    sorted_pts = sorted(points, key=lambda p: p["mean_acceptance_length"])
    frontier = []
    best_speedup = -1.0
    for pt in sorted_pts:
        if pt["speedup"] >= best_speedup:
            frontier.append(pt)
            best_speedup = pt["speedup"]
    return frontier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RAW_JSONL)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    if not args.results.exists():
        print(f"No results file found at {args.results}. Run the benchmark first.")
        return

    records = load_records(args.results)
    if not records:
        print("Results file is empty.")
        return

    # Drop EAGLE measurements taken with a superseded decoding path. They are
    # real measurements of a different implementation, so averaging them in
    # with current ones would produce a number that describes neither.
    from specbench.eagle.spec_decode_eagle import INFERENCE_VERSION

    before = len(records)
    records = [
        r
        for r in records
        if r.get("mode") != "spec_eagle" or r.get("inference_version") == INFERENCE_VERSION
    ]
    dropped = before - len(records)
    if dropped:
        print(
            f"Excluded {dropped} EAGLE records measured with a decoding path older "
            f"than inference v{INFERENCE_VERSION}."
        )

    # Group by (mode, draft_model, K, temperature). Temperature belongs in the
    # key: a greedy baseline and a sampling baseline are different conditions,
    # and pooling them would compare each speculative arm against a mixture of
    # both. Baselines are likewise selected per temperature so that every
    # comparison is paired within its own condition.
    groups = defaultdict(list)
    for r in records:
        key = (
            r.get("mode", ""),
            r.get("draft_model", ""),
            r.get("K", 0),
            r.get("temperature", 1.0),
        )
        groups[key].append(r)

    baselines_by_temp = defaultdict(list)
    for r in records:
        if r.get("mode") == "baseline":
            baselines_by_temp[r.get("temperature", 1.0)].append(r)

    rows = []
    for (mode, draft_model, K, temperature), group_records in groups.items():
        if mode == "baseline":
            continue
        combined = baselines_by_temp.get(temperature, []) + group_records
        agg = aggregate_config(combined)
        if agg:
            agg["temperature"] = temperature
            rows.append(agg)

    if not rows:
        print("Not enough data to produce Pareto curve. Run more benchmarks first.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(PARETO_CSV, index=False)
    print(f"Pareto data written to {PARETO_CSV}")
    print(df[["mode", "K", "temperature", "draft_model", "mean_acceptance_length",
              "speedup", "speedup_ci_lo", "speedup_ci_hi"]].to_string(index=False))

    # Produce the plot.
    try:
        import matplotlib

        matplotlib.use("Agg")  # No display available in a batch run.
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 7))

        # Color by mode.
        mode_colors = {
            "spec": "#2196F3",        # Blue for vanilla spec
            "spec_eagle": "#FF5722",  # Orange for EAGLE variants
        }

        points_for_pareto = []

        for _, row in df.iterrows():
            color = mode_colors.get(row["mode"], "#9E9E9E")
            label = f"{row['draft_model']} K={int(row['K'])}"

            ax.errorbar(
                row["mean_acceptance_length"],
                row["speedup"],
                xerr=[[row["mean_acceptance_length"] - row["acceptance_length_ci_lo"]],
                      [row["acceptance_length_ci_hi"] - row["mean_acceptance_length"]]],
                yerr=[[row["speedup"] - row["speedup_ci_lo"]],
                      [row["speedup_ci_hi"] - row["speedup"]]],
                fmt="o",
                color=color,
                markersize=8,
                capsize=4,
                alpha=0.8,
            )
            ax.annotate(
                label,
                (row["mean_acceptance_length"], row["speedup"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.7,
            )
            points_for_pareto.append({
                "mean_acceptance_length": row["mean_acceptance_length"],
                "speedup": row["speedup"],
            })

        # Mark Pareto frontier.
        frontier = pareto_frontier(points_for_pareto)
        if len(frontier) >= 2:
            fx = [p["mean_acceptance_length"] for p in frontier]
            fy = [p["speedup"] for p in frontier]
            ax.plot(fx, fy, "k--", linewidth=1.5, alpha=0.5, label="Pareto frontier")

        # Add baseline reference line.
        ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.5, label="Baseline (1.0×)")

        # Legend.
        legend_patches = [
            mpatches.Patch(color=mode_colors["spec"], label="Vanilla spec (Phase 1)"),
            mpatches.Patch(color=mode_colors["spec_eagle"], label="EAGLE head (Phase 2)"),
        ]
        ax.legend(handles=legend_patches + [
            plt.Line2D([0], [0], color="k", linestyle="--", label="Pareto frontier"),
        ], loc="upper left", fontsize=9)

        ax.set_xlabel("Mean acceptance length (tokens per target forward pass)", fontsize=11)
        ax.set_ylabel("Wall-clock speedup vs. baseline (×)", fontsize=11)
        ax.set_title(
            "Speculative Decoding: Acceptance Length vs. Speedup\n"
            "GTX 1650 Ti Mobile, TinyLlama-1.1B target, 95% CI error bars",
            fontsize=12,
        )
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.9)

        out_path = FIGURES_DIR / "pareto_acceptance_vs_speedup.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Pareto plot saved to {out_path}")

    except Exception as e:
        print(f"Warning: could not produce Pareto plot: {e}")


if __name__ == "__main__":
    main()
