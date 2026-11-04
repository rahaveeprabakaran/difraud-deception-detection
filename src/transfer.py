"""Cross-domain transfer: train on one domain, score the other domains' TEST sets once.

Rules that keep the numbers clean:
- The model is trained on the source domain's train split with fixed settings (no tuning).
- Target test texts are only scored. Nothing (threshold, epochs, model) is chosen with them.
- Only labels and scores are saved, never texts. The number of target test texts that also occur
  in the source train split is saved as indices, so overlap can be excluded in the analysis.

Usage after training:  python -m src.transfer results_transfer
Writes results_transfer/transfer.md and transfer_auc.png.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.data import DOMAINS, get_path, read_jsonl

MAX_CHARS = 20_000


def score_targets(source, train_texts, manifest, predict_probs, domains=DOMAINS):
    """predict_probs(texts, labels) -> P(deceptive) per text. Returns labels and scores only."""
    train_set = set(train_texts)
    out = {}
    for t in domains:
        if t == source:
            continue
        texts, labels = read_jsonl(get_path(t, "test", manifest))
        probs = predict_probs([x[:MAX_CHARS] for x in texts], labels)
        out[t] = {"y": [int(v) for v in labels],
                  "p1": [round(float(p), 5) for p in probs],
                  "overlap_idx": [i for i, x in enumerate(texts) if x in train_set]}
    return out


def _metrics(y, p, keep=None):
    y, p = np.asarray(y), np.asarray(p)
    if keep is not None:
        y, p = y[keep], p[keep]
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)),
            "trivial": float(f1_score(y, np.ones_like(y), zero_division=0)), "n": int(len(y))}


def build(folder):
    cells, overlap, val_f1 = {}, {}, {}
    for fp in sorted(Path(folder).glob("*.json")):
        r = json.loads(fp.read_text())
        if "transfer" not in r:
            continue
        src = r["meta"]["domain"]
        val_f1[src] = r["validation"]["f1"]
        t = r["predictions"]["test"]
        cells[(src, src)] = _metrics(t["y"], t["p1"])
        for tgt, d in r["transfer"].items():
            cells[(src, tgt)] = _metrics(d["y"], d["p1"])
            if d["overlap_idx"]:
                keep = np.ones(len(d["y"]), dtype=bool)
                keep[d["overlap_idx"]] = False
                overlap[(src, tgt)] = (len(d["overlap_idx"]), _metrics(d["y"], d["p1"], keep))
    return cells, overlap, val_f1


def _matrix(cells, key, fmt, srcs, tgts):
    rows = ["| Train on \\ Test on | " + " | ".join(tgts) + " |", "|---|" + "---|" * len(tgts)]
    for s in srcs:
        vals = []
        for t in tgts:
            v = cells.get((s, t))
            txt = "" if v is None else fmt.format(v[key])
            vals.append(f"**{txt}**" if s == t and txt else txt)
        rows.append(f"| {s} | " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main(folder):
    cells, overlap, val_f1 = build(folder)
    srcs = [d for d in DOMAINS if any(k[0] == d for k in cells)]
    tgts = list(DOMAINS)
    parts = ["# Cross-domain transfer (one BERT-base run per source domain, seed 42, fixed settings)\n",
             "Rows = training domain, columns = test domain. Diagonal (bold) = in-domain test. "
             "AUC does not depend on a threshold or on how common deception is; 0.5 is chance.\n",
             "## ROC AUC\n", _matrix(cells, "auc", "{:.3f}", srcs, tgts), "\n",
             "## Binary F1 at threshold 0.5 (deceptive = positive)\n", _matrix(cells, "f1", "{:.3f}", srcs, tgts), "\n",
             "Trivial all-deceptive F1 per test domain: " + ", ".join(
                 f"{t} {cells[(srcs[0], t)]['trivial']:.3f}" for t in tgts if (srcs[0], t) in cells) + "\n"]
    if overlap:
        parts.append("## Texts shared between the training domain and the test domain\n")
        parts.append("| Train on | Test on | Shared texts | AUC without them |\n|---|---|---|---|")
        for (s, t), (n, m) in sorted(overlap.items()):
            parts.append(f"| {s} | {t} | {n} | {m['auc']:.3f} |")
    else:
        parts.append("No identical text occurs in both a training domain and a different test domain.\n")
    Path(folder, "transfer.md").write_text("\n".join(parts) + "\n")
    print("\n".join(parts))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mat = np.array([[cells.get((s, t), {"auc": np.nan})["auc"] for t in tgts] for s in srcs])
        fig, ax = plt.subplots(figsize=(8, 4.8))
        im = ax.imshow(mat, vmin=0.4, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(tgts)), [t.replace("_", "\n") for t in tgts], fontsize=8)
        ax.set_yticks(range(len(srcs)), srcs, fontsize=8)
        for i in range(len(srcs)):
            for j in range(len(tgts)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if mat[i, j] < 0.75 else "black")
        ax.set_xlabel("Test domain"); ax.set_ylabel("Training domain")
        ax.set_title("Cross-domain transfer, ROC AUC (0.5 = chance)")
        fig.colorbar(im, ax=ax); fig.tight_layout()
        fig.savefig(Path(folder, "transfer_auc.png"), dpi=160)
    except Exception as e:  # figure is optional
        print("figure skipped:", e)


if __name__ == "__main__":
    main(sys.argv[1])
