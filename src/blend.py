"""Blend test: does mixing the transformer with TF-IDF help? (CPU only)

Usage: python -m src.blend results_pred
For each domain: p = w * transformer + (1 - w) * TF-IDF. The weight w (0, 0.1, ..., 1) is chosen on
VALIDATION F1 at threshold 0.5 only. If several weights tie, the one closest to w = 1 (transformer only) wins.
The test set is scored once, with that weight. A paired bootstrap gives the 95% interval of
F1(blend) - F1(transformer ensemble). Whatever the result is, it is reported.
Only labels and scores are used, never texts. TF-IDF settings are identical to src/tfidf.py.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from src.analyze import load
from src.data import get_path, load_manifest, read_jsonl
from src.paired import FINAL, MAX_CHARS, paired_diff

WEIGHTS = np.round(np.linspace(0.0, 1.0, 11), 2)


def blend(p_trans, p_tfidf, w):
    return w * np.asarray(p_trans) + (1.0 - w) * np.asarray(p_tfidf)


def choose_weight(y, p_trans, p_tfidf, weights=WEIGHTS):
    """Weight with the highest F1@0.5 on the given (validation) data; ties go to the larger weight."""
    f = [f1_score(y, (blend(p_trans, p_tfidf, w) >= 0.5).astype(int), zero_division=0) for w in weights]
    best = max(f)
    w = max(w for w, v in zip(weights, f) if v == best)
    return float(w), float(best)


def tfidf_val_test(domain, manifest):
    sp = {s: read_jsonl(get_path(domain, s, manifest)) for s in ("train", "validation", "test")}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
    A = vec.fit_transform([t[:MAX_CHARS] for t in sp["train"][0]])
    clf = LogisticRegression(C=4.0, max_iter=2000).fit(A, sp["train"][1])
    out = {}
    for s in ("validation", "test"):
        out[s] = (np.array(sp[s][1]), clf.predict_proba(vec.transform([t[:MAX_CHARS] for t in sp[s][0]]))[:, 1])
    return out


def main(pred_dir):
    runs, manifest = load(pred_dir), load_manifest()
    rows = ["| Domain | Weight on transformer (chosen on validation) | Val F1 blend | Test F1 transformer | "
            "Test F1 blend | Difference | 95% CI | AUC transformer | AUC blend |", "|---|---|---|---|---|---|---|---|---|"]
    for d, m in FINAL.items():
        rs = runs[(d, m)]
        tf = tfidf_val_test(d, manifest)
        yv, pv_t = np.array(rs[0]["validation"]["y"]), np.mean([r["validation"]["p1"] for r in rs], axis=0)
        yt, pt_t = np.array(rs[0]["test"]["y"]), np.mean([r["test"]["p1"] for r in rs], axis=0)
        assert (yv == tf["validation"][0]).all() and (yt == tf["test"][0]).all(), f"{d}: labels differ"
        w, fv = choose_weight(yv, pv_t, tf["validation"][1])
        pt_b = blend(pt_t, tf["test"][1], w)
        diff, lo, hi = paired_diff(yt, pt_b, pt_t)
        f_t = f1_score(yt, (pt_t >= .5).astype(int), zero_division=0)
        f_b = f1_score(yt, (pt_b >= .5).astype(int), zero_division=0)
        rows.append(f"| {d} | {w:.1f} | {fv:.4f} | {f_t:.4f} | {f_b:.4f} | {diff:+.4f} | [{lo:+.3f}, {hi:+.3f}] | "
                    f"{roc_auc_score(yt, pt_t):.3f} | {roc_auc_score(yt, pt_b):.3f} |")
    out = "\n".join(rows)
    print(out)
    Path(pred_dir, "blend.md").write_text(out + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
