"""Summary from saved predictions: single runs, seed ensemble, bootstrap 95% CIs.

Usage: python -m src.analyze results_pred
Uses only labels and scores. Thresholds are chosen on validation, then applied once to test.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.threshold import best_threshold, f1_at

# Paper's plain fine-tuning row ("finetune, tgt"), LREC-COLING 2024 Table 2.
PAPER_FT = {"political_statements": 0.7640, "fake_news": 0.9773, "job_scams": 0.6729,
            "product_reviews": 0.6968, "phishing": 0.9859, "twitter_rumours": 0.7944,
            "sms": 0.9883}


def load(folder):
    runs = defaultdict(list)
    for fp in sorted(Path(folder).glob("*.json")):
        r = json.loads(fp.read_text())
        if "predictions" in r:
            m = r["meta"]
            runs[(m["domain"], m["model"])].append(r["predictions"])
    return runs


def boot_ci(y, p, thr, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pred = (np.asarray(p) >= thr).astype(int)
    m = len(y)
    vals = []
    for _ in range(n):
        i = rng.integers(0, m, m)
        vals.append(f1_score(y[i], pred[i], pos_label=1, zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main(folder):
    rows = ["| Domain | Model | Runs | Single runs @0.5 (mean ± std) | Single runs @tuned thr | "
            "Ensemble @0.5 [95% CI] | Ensemble @tuned thr [95% CI] | Trivial F1 | AUC | AP | Paper fine-tune |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (dom, model), runs in sorted(load(folder).items()):
        yv, yt = np.array(runs[0]["validation"]["y"]), np.array(runs[0]["test"]["y"])
        for r in runs:
            assert np.array_equal(yt, np.array(r["test"]["y"])), "test order differs"
        pv = [np.array(r["validation"]["p1"]) for r in runs]
        pt = [np.array(r["test"]["p1"]) for r in runs]
        s05 = [f1_at(yt, p, 0.5) for p in pt]
        stuned = []
        for a, b in zip(pv, pt):
            thr, _ = best_threshold(yv, a)
            stuned.append(f1_at(yt, b, thr))
        ev, et = np.mean(pv, axis=0), np.mean(pt, axis=0)
        thr, _ = best_threshold(yv, ev)
        e05, etn = f1_at(yt, et, 0.5), f1_at(yt, et, thr)
        c05, ctn = boot_ci(yt, et, 0.5), boot_ci(yt, et, thr)
        sd = f"{np.std(s05, ddof=1):.3f}" if len(s05) > 1 else "n/a"
        rows.append(f"| {dom} | {model.split('/')[-1]} | {len(runs)} | {np.mean(s05):.4f} ± {sd} | "
                    f"{np.mean(stuned):.4f} | {e05:.4f} [{c05[0]:.3f}, {c05[1]:.3f}] | "
                    f"{etn:.4f} [{ctn[0]:.3f}, {ctn[1]:.3f}] (thr {thr:.2f}) | "
                    f"{f1_score(yt, np.ones_like(yt), zero_division=0):.3f} | {roc_auc_score(yt, et):.3f} | "
                    f"{average_precision_score(yt, et):.3f} | {PAPER_FT.get(dom, '')} |")
    out = "\n".join(rows)
    Path(folder, "analysis.md").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_pred")
