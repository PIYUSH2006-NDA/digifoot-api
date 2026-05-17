"""
summarize_validation.py
Aggregate per-pair reports into one summary.
"""
import json, sys, numpy as np
from pathlib import Path


def main(root):
    root = Path(root)
    reps = list(root.glob("pair_*/report.json"))
    if not reps: raise SystemExit("no reports found")
    keys = ["mean_mm", "median_mm", "p95_mm", "max_mm",
            "f_at_1mm", "f_at_2mm", "f_at_3mm"]
    agg = {k: [] for k in keys}
    for r in reps:
        d = json.loads(r.read_text())
        for k in keys: agg[k].append(d[k])
    out = {f"{k}_mean": float(np.mean(v)) for k, v in agg.items()}
    out["n_pairs"] = len(reps)
    out["pass_median_1mm"] = bool(out["median_mm_mean"] <= 1.0)
    out["pass_p95_2mm"] = bool(out["p95_mm_mean"] <= 2.0)
    out["pass_f1mm_85"] = bool(out["f_at_1mm_mean"] >= 0.85)
    print(json.dumps(out, indent=2))
    (root / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "validation_set")
