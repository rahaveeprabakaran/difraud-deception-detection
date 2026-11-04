"""Characterise repeated texts in one domain (counts only, no text is printed).

Questions answered:
  1. Test texts that also appear in train: do the labels agree?
  2. Texts repeated inside train: do their labels agree?
  3. How many more repeats appear if matching ignores case, spacing and links?

Usage: python -m src.overlap --domain phishing
"""
import argparse
import re
from collections import defaultdict

from src.data import DOMAINS, get_path, load_manifest, read_jsonl


def norm(t):
    t = re.sub(r"https?://\S+", "URL", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def _groups(texts, labels, key):
    g = defaultdict(list)
    for t, y in zip(texts, labels):
        g[key(t)].append(y)
    return g


def analyze(train_x, train_y, test_x, test_y):
    out = {}
    for name, key in (("exact", lambda t: t), ("normalized", norm)):
        tr = _groups(train_x, train_y, key)
        within = [v for v in tr.values() if len(v) > 1]
        shared = [(key(t), y) for t, y in zip(test_x, test_y) if key(t) in tr]
        out[name] = {
            "test_in_train": len(shared),
            "test_in_train_label_agrees": sum(all(y == l for l in tr[k]) for k, y in shared),
            "test_in_train_label_conflict": sum(any(y != l for l in tr[k]) for k, y in shared),
            "train_extra_copies": sum(len(v) - 1 for v in within),
            "train_groups_label_conflict": sum(len(set(v)) > 1 for v in within),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    a = ap.parse_args()
    m = load_manifest()
    trx, try_ = read_jsonl(get_path(a.domain, "train", m))
    tex, tey = read_jsonl(get_path(a.domain, "test", m))
    res = analyze(trx, try_, tex, tey)
    print(f"{a.domain}: train {len(trx)}, test {len(tex)}")
    for k, v in res.items():
        print(f"[{k} match]")
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")


if __name__ == "__main__":
    main()
