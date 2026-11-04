"""How much do test texts that also appear in train change the score?

Usage: python -m src.dupes --domain phishing --run results_pred/phishing__bert-base-uncased__s42.json
"""
import argparse
import json

import numpy as np

from src.data import DOMAINS, get_path, load_manifest, read_jsonl
from src.threshold import f1_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    ap.add_argument("--run", required=True)
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    manifest = load_manifest()
    train, _ = read_jsonl(get_path(a.domain, "train", manifest))
    test, labels = read_jsonl(get_path(a.domain, "test", manifest))
    pr = json.loads(open(a.run).read())["predictions"]["test"]
    y, p = np.array(pr["y"]), np.array(pr["p1"])
    assert list(y) == labels, "run does not match this domain's test file"
    seen = set(train)
    keep = np.array([t not in seen for t in test])
    print(f"test texts: {len(test)}; also in train: {int((~keep).sum())}")
    print(f"F1 all test:           {f1_at(y, p, a.thr):.4f}")
    print(f"F1 without the overlap: {f1_at(y[keep], p[keep], a.thr):.4f}")


if __name__ == "__main__":
    main()
