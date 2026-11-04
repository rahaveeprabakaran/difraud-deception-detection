"""Paired bootstrap: is the transformer's test F1 really different from TF-IDF's? (threshold 0.5 for both)

Usage: python -m src.paired results_pred
Both models are scored on the same test texts and the same resamples; the 95% interval is for
F1(transformer) - F1(TF-IDF). If it contains 0, no real difference can be claimed.
It covers test-sampling noise only (not training randomness). TF-IDF settings are identical to src/tfidf.py.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.analyze import load
from src.data import get_path, load_manifest, read_jsonl

MAX_CHARS = 20_000
FINAL = {"sms": "google-bert/bert-base-uncased", "phishing": "google-bert/bert-base-uncased",
         "fake_news": "google-bert/bert-base-uncased", "product_reviews": "google-bert/bert-base-uncased",
         "political_statements": "google-bert/bert-base-uncased",
         "twitter_rumours": "FacebookAI/roberta-base", "job_scams": "FacebookAI/roberta-base"}


def paired_diff(y, p_a, p_b, thr=0.5, n=2000, seed=0):
    y = np.asarray(y)
    a = (np.asarray(p_a) >= thr).astype(int)
    b = (np.asarray(p_b) >= thr).astype(int)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(f1_score(y[i], a[i], zero_division=0) - f1_score(y[i], b[i], zero_division=0))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return (f1_score(y, a, zero_division=0) - f1_score(y, b, zero_division=0)), lo, hi


def tfidf_scores(domain, manifest):
    sp = {s: read_jsonl(get_path(domain, s, manifest)) for s in ("train", "test")}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
    A = vec.fit_transform([t[:MAX_CHARS] for t in sp["train"][0]])
    clf = LogisticRegression(C=4.0, max_iter=2000).fit(A, sp["train"][1])
    return np.array(sp["test"][1]), clf.predict_proba(vec.transform([t[:MAX_CHARS] for t in sp["test"][0]]))[:, 1]


def main(pred_dir):
    runs, manifest = load(pred_dir), load_manifest()
    rows = ["| Domain | Transformer F1 | TF-IDF F1 | Difference | 95% CI of difference | Real difference? |",
            "|---|---|---|---|---|---|"]
    for d, m in FINAL.items():
        rs = runs[(d, m)]
        y_t = np.array(rs[0]["test"]["y"])
        p_t = np.mean([r["test"]["p1"] for r in rs], axis=0)
        y, p_f = tfidf_scores(d, manifest)
        assert (y == y_t).all(), f"{d}: test labels differ between the two models"
        diff, lo, hi = paired_diff(y, p_t, p_f)
        f_t = f1_score(y, (p_t >= .5).astype(int)); f_f = f1_score(y, (p_f >= .5).astype(int))
        verdict = "yes (transformer better)" if lo > 0 else "yes (TF-IDF better)" if hi < 0 else "no (interval contains 0)"
        rows.append(f"| {d} | {f_t:.4f} | {f_f:.4f} | {diff:+.4f} | [{lo:+.3f}, {hi:+.3f}] | {verdict} |")
    out = "\n".join(rows)
    print(out)
    Path(pred_dir, "paired.md").write_text(out + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
