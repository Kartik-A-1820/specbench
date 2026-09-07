"""
scripts/report_status.py — print what actually completed.

Run at the end of run_all.bat. Reports only what is on disk. A stage that did
not finish shows as MISSING; nothing is inferred, estimated or filled in.

Usage:
    python scripts/report_status.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results"


def _load(path: Path):
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        return None


def main() -> int:
    print("\n================ SPECBENCH STATUS ================")

    probe = _load(RESULTS / "vram_probe.json")
    if probe:
        m = probe.get("measured", {})
        print(
            f"VRAM probe:        peak {m.get('peak_allocated_gb', 0):.3f} GB, "
            f"within budget = {probe.get('within_budget')}"
        )
    else:
        print("VRAM probe:        MISSING")

    raw = RESULTS / "raw_measurements.jsonl"
    if raw.exists():
        n = sum(1 for _ in raw.open(encoding="utf-8"))
        print(f"Raw measurements:  {n} run records")
    else:
        print("Raw measurements:  MISSING")

    summary = _load(RESULTS / "summary.json")
    if not summary:
        print("Summary:           MISSING — no benchmark completed.")
        print("=================================================\n")
        return 0

    print(f"Summary:           {len(summary)} configurations\n")
    header = f"{'configuration':44s} {'tok/s':>8s} {'speedup':>9s} {'95% CI':>16s} {'alpha':>7s}"
    print(header)
    print("-" * len(header))
    for name in sorted(summary):
        s = summary[name]
        if not isinstance(s, dict) or "mean_tps" not in s:
            continue
        ci = f"{s.get('speedup_ci_95_low', 0):.2f}-{s.get('speedup_ci_95_high', 0):.2f}"
        print(
            f"{name:44s} {s['mean_tps']:8.2f} {s.get('speedup', 0):9.3f} "
            f"{ci:>16s} {s.get('mean_acceptance_length', 0):7.3f}"
        )

    figures = sorted((RESULTS / "figures").glob("*.png")) if (RESULTS / "figures").exists() else []
    print(f"\nFigures:           {len(figures)} " + (f"({', '.join(f.name for f in figures)})" if figures else "MISSING"))

    ckpts = _load(RESULTS / "eagle_checkpoints.json")
    print(f"EAGLE heads:       {len(ckpts) if ckpts else 0} trained")

    print("=================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
