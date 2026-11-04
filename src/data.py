"""Download, pin, verify and validate DIFrauD. Treats every text as untrusted input.

Safety choices:
- Downloads only the plain .jsonl files. It never runs the repo's difraud.py loading script.
- Pins the dataset to one commit and records a SHA-256 for each file (data_manifest.json).
- Texts are only parsed with json.loads and tokenized. They are never executed, opened as
  links, or rendered as HTML.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = "difraud/difraud"
DOMAINS = ["fake_news", "job_scams", "phishing", "political_statements",
           "product_reviews", "sms", "twitter_rumours"]
SPLITS = ("train", "validation", "test")
MANIFEST = Path("data_manifest.json")
MAX_LINE_CHARS = 50_000_000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pin(manifest_path=MANIFEST):
    """Run once. Resolves the current dataset commit and records file hashes."""
    from huggingface_hub import HfApi, hf_hub_download
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    revision = HfApi().dataset_info(REPO).sha
    files = {}
    for d in DOMAINS:
        for s in SPLITS:
            name = f"{d}/{s}.jsonl"
            p = hf_hub_download(REPO, name, repo_type="dataset", revision=revision)
            files[name] = sha256(p)
    Path(manifest_path).write_text(json.dumps(
        {"repo": REPO, "revision": revision, "files": files}, indent=2))
    print(f"Pinned {REPO} at {revision}. Wrote {manifest_path}. Commit this file.")


def load_manifest(manifest_path=MANIFEST):
    if not Path(manifest_path).exists():
        sys.exit("data_manifest.json not found. Run: python -m src.data pin")
    return json.loads(Path(manifest_path).read_text())


def get_path(domain, split, manifest):
    from huggingface_hub import hf_hub_download
    name = f"{domain}/{split}.jsonl"
    p = hf_hub_download(manifest["repo"], name, repo_type="dataset",
                        revision=manifest["revision"])
    if sha256(p) != manifest["files"][name]:
        raise RuntimeError(f"Hash mismatch for {name}. Do not use this file.")
    return p


def read_jsonl(path):
    """Return (texts, labels). Fails loudly on bad schema, labels or empty text."""
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if len(line) > MAX_LINE_CHARS:
                raise ValueError(f"{path}:{i} line too long")
            r = json.loads(line)
            t, y = r.get("text"), r.get("label")
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"{path}:{i} empty or non-string text")
            if y not in (0, 1) or isinstance(y, bool):
                raise ValueError(f"{path}:{i} label must be 0 or 1, got {y!r}")
            texts.append(t)
            labels.append(int(y))
    return texts, labels


def check_splits(splits):
    """splits: {'train': (texts, labels), ...}. Returns a report; flags leakage."""
    rep = {}
    seen = {s: set(splits[s][0]) for s in splits}
    for s, (t, y) in splits.items():
        n = len(y)
        rep[s] = {"n": n, "deceptive": sum(y), "genuine": n - sum(y),
                  "dup_within": n - len(seen[s])}
    rep["leak_train_test"] = len(seen["train"] & seen["test"])
    rep["leak_train_val"] = len(seen["train"] & seen["validation"])
    rep["leak_val_test"] = len(seen["validation"] & seen["test"])
    return rep


def load_domain(domain, manifest):
    splits = {s: read_jsonl(get_path(domain, s, manifest)) for s in SPLITS}
    for s, (t, y) in splits.items():
        if len(set(y)) < 2:
            raise ValueError(f"{domain}/{s} contains only one class")
    return splits, check_splits(splits)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "pin":
        pin()
    else:
        sys.exit("usage: python -m src.data pin")
