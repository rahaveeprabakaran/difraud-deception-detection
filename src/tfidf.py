"""Simple baseline: TF-IDF + logistic regression on every domain (CPU only).

Usage: python -m src.tfidf
Prints a table and the most indicative words per class. Only words are printed, never texts.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.data import DOMAINS, get_path, load_manifest, read_jsonl
from src.threshold import best_threshold

MAX_CHARS = 20_000


def run_domain(tr, va, te, top=12):
    (xtr, ytr), (xva, yva), (xte, yte) = tr, va, te
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
    A = vec.fit_transform([t[:MAX_CHARS] for t in xtr])
    clf = LogisticRegression(C=4.0, max_iter=2000).fit(A, ytr)
    pv = clf.predict_proba(vec.transform([t[:MAX_CHARS] for t in xva]))[:, 1]
    pt = clf.predict_proba(vec.transform([t[:MAX_CHARS] for t in xte]))[:, 1]
    thr, _ = best_threshold(np.array(yva), pv)
    yt = np.array(yte)
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    return {"f1_05": f1_score(yt, (pt >= 0.5).astype(int), zero_division=0),
            "f1_tuned": f1_score(yt, (pt >= thr).astype(int), zero_division=0), "thr": thr,
            "auc": roc_auc_score(yt, pt), "ap": average_precision_score(yt, pt),
            "trivial": f1_score(yt, np.ones_like(yt), zero_division=0),
            "words_deceptive": names[order[-top:]][::-1].tolist(),
            "words_genuine": names[order[:top]].tolist()}


def main():
    manifest = load_manifest()
    rows = ["| Domain | F1 @0.5 | F1 @tuned thr | Trivial F1 | AUC | AP |", "|---|---|---|---|---|---|"]
    out = {}
    for d in DOMAINS:
        sp = [read_jsonl(get_path(d, s, manifest)) for s in ("train", "validation", "test")]
        r = run_domain(*[(x[0], x[1]) for x in sp])
        out[d] = r
        rows.append(f"| {d} | {r['f1_05']:.4f} | {r['f1_tuned']:.4f} | {r['trivial']:.3f} | {r['auc']:.3f} | {r['ap']:.3f} |")
    print("\n".join(rows))
    for d, r in out.items():
        print(f"\n{d}\n  deceptive words: {', '.join(r['words_deceptive'])}\n  genuine words:   {', '.join(r['words_genuine'])}")
    Path("results_tfidf").mkdir(exist_ok=True)
    Path("results_tfidf/tfidf.json").write_text(json.dumps(out, indent=2))
    Path("results_tfidf/tfidf.md").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
