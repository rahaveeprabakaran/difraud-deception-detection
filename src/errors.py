"""Print misclassified test texts for one run, as inert plain text.

Usage: python -m src.errors --domain job_scams --run results_pred/job_scams__roberta-base__s42.json --kind fn --n 25
The texts are scams or phishing. They are only printed, never opened. Links are defanged
(http -> hxxp). Do not click or paste anything shown here into a browser.
"""
import argparse
import json
import re

import numpy as np

from src.data import DOMAINS, get_path, load_manifest, read_jsonl


def clean(t, n=300):
    t = re.sub(r"[^\x20-\x7E]", " ", t)
    t = re.sub(r"https?://", lambda m: m.group(0).replace("http", "hxxp"), t)
    return re.sub(r"\s+", " ", t)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    ap.add_argument("--run", required=True)
    ap.add_argument("--kind", choices=["fn", "fp"], default="fn",
                    help="fn = deceptive texts the model missed; fp = genuine texts it flagged")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=25)
    a = ap.parse_args()
    manifest = load_manifest()
    texts, labels = read_jsonl(get_path(a.domain, "test", manifest))
    pr = json.loads(open(a.run).read())["predictions"]["test"]
    y, p = np.array(pr["y"]), np.array(pr["p1"])
    assert list(y) == labels, "run does not match this domain's test file"
    pred = (p >= a.thr).astype(int)
    idx = np.where((y == 1) & (pred == 0))[0] if a.kind == "fn" else np.where((y == 0) & (pred == 1))[0]
    print(f"{len(idx)} {a.kind} at threshold {a.thr}. Showing {min(a.n, len(idx))} (lowest-confidence first).")
    order = idx[np.argsort(p[idx])] if a.kind == "fn" else idx[np.argsort(-p[idx])]
    for i in order[: a.n]:
        print(f"[{i}] p(deceptive)={p[i]:.2f} | {clean(texts[i])}")


if __name__ == "__main__":
    main()
