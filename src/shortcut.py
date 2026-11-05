"""Shortcut test: does the simple TF-IDF model lose accuracy when its top "deceptive" words are removed?

Usage: python -m src.shortcut --domain political_statements --k 20

Steps (CPU only, takes about a minute):
  1. Fit TF-IDF + logistic regression (same settings as src/tfidf.py) and score the test set.
  2. Remove the k single words with the largest "deceptive" weights from ALL texts (train, validation, test), refit, score.
  3. Control: remove k other words of similar frequency, chosen at random (5 repeats), refit, score.
If removing the top words hurts clearly more than removing random words, the model relied on them.
Only words and numbers are printed, never texts.
"""
import argparse
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from src.data import DOMAINS, get_path, load_manifest, read_jsonl

MAX_CHARS = 20_000


def strip_words(texts, words):
    """Remove whole words (case-insensitive) from every text."""
    if not words:
        return list(texts)
    pat = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in sorted(words)) + r")\b", re.IGNORECASE)
    return [pat.sub(" ", t) for t in texts]


def fit_score(xtr, ytr, xte, yte):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)
    A = vec.fit_transform([t[:MAX_CHARS] for t in xtr])
    clf = LogisticRegression(C=4.0, max_iter=2000).fit(A, ytr)
    p = clf.predict_proba(vec.transform([t[:MAX_CHARS] for t in xte]))[:, 1]
    yte = np.asarray(yte)
    f1 = f1_score(yte, (p >= 0.5).astype(int), zero_division=0)
    return f1, roc_auc_score(yte, p), vec, clf, A


def top_unigrams(vec, clf, k):
    names = np.array(vec.get_feature_names_out())
    uni = np.array([" " not in n for n in names])
    idx = np.where(uni)[0]
    order = idx[np.argsort(clf.coef_[0][idx])[::-1]]
    return names[order[:k]].tolist()


def matched_random_words(vec, A, remove, k, rng):
    """k random unigrams whose document frequency is within a factor of 2 of the removed words."""
    names = np.array(vec.get_feature_names_out())
    df = np.asarray((A > 0).sum(axis=0)).ravel()
    pos = {n: i for i, n in enumerate(names)}
    lo = 0.5 * min(df[pos[w]] for w in remove)
    hi = 2.0 * max(df[pos[w]] for w in remove)
    drop = set(remove)
    ok = [i for i, n in enumerate(names) if " " not in n and lo <= df[i] <= hi and n not in drop]
    if not ok:
        raise ValueError("no control words with similar frequency")
    pick = rng.choice(np.array(ok), size=min(k, len(ok)), replace=False)
    return names[pick].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    m = load_manifest()
    xtr, ytr = read_jsonl(get_path(a.domain, "train", m))
    xte, yte = read_jsonl(get_path(a.domain, "test", m))
    f1, auc, vec, clf, A = fit_score(xtr, ytr, xte, yte)
    words = top_unigrams(vec, clf, a.k)
    print(f"{a.domain}: train {len(xtr)}, test {len(xte)}")
    print(f"Baseline: F1@0.5 {f1:.4f}, AUC {auc:.3f}")
    print(f"Top {a.k} deceptive words removed: {', '.join(words)}")
    f1r, aucr, *_ = fit_score(strip_words(xtr, words), ytr, strip_words(xte, words), yte)
    print(f"Without them: F1@0.5 {f1r:.4f} ({f1r - f1:+.4f}), AUC {aucr:.3f} ({aucr - auc:+.3f})")
    rng = np.random.default_rng(0)
    rf, ra = [], []
    for _ in range(a.repeats):
        ctrl = matched_random_words(vec, A, words, a.k, rng)
        f, au, *_ = fit_score(strip_words(xtr, ctrl), ytr, strip_words(xte, ctrl), yte)
        rf.append(f)
        ra.append(au)
    print(f"Control, {a.k} random words of similar frequency removed ({a.repeats} repeats): "
          f"F1@0.5 {np.mean(rf):.4f} ({np.mean(rf) - f1:+.4f}, range {min(rf):.4f} to {max(rf):.4f}), "
          f"AUC {np.mean(ra):.3f} ({np.mean(ra) - auc:+.3f})")


if __name__ == "__main__":
    main()
