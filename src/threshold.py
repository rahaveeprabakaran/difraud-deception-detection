"""Pick the decision threshold on validation, then apply it once to test.

Reads the saved prediction scores (no texts). Usage:
  python -m src.threshold results_us_e5 results_us_e10
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

GRID = np.round(np.linspace(0.01, 0.99, 99), 3)


def f1_at(y, p, t):
    return f1_score(y, (p >= t).astype(int), pos_label=1, zero_division=0)


def best_threshold(y, p):
    f = np.array([f1_at(y, p, t) for t in GRID])
    ties = np.where(f == f.max())[0]
    return float(GRID[ties[len(ties) // 2]]), float(f.max())  # middle of tied thresholds


def main(folders):
    print("file | best val threshold | val F1 at it | test F1 at 0.5 | test F1 at it")
    for folder in folders:
        for fp in sorted(Path(folder).glob("*.json")):
            r = json.loads(fp.read_text())
            if "predictions" not in r:
                continue
            v, t = r["predictions"]["validation"], r["predictions"]["test"]
            yv, pv = np.array(v["y"]), np.array(v["p1"])
            yt, pt = np.array(t["y"]), np.array(t["p1"])
            thr, fv = best_threshold(yv, pv)
            print(f"{folder}/{fp.name} | {thr:.2f} | {fv:.4f} | {f1_at(yt, pt, 0.5):.4f} | {f1_at(yt, pt, thr):.4f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results"])
