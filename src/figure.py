"""One summary figure: test F1 per domain for trivial, TF-IDF, transformer, and the paper's fine-tune.

Usage: python -m src.figure results_pred results_tfidf/tfidf.json
Writes summary.png. Reads only labels, scores and numbers (no texts).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from src.analyze import PAPER_FT, load

# Final model per domain (same choice as the README table).
FINAL = {"sms": "google-bert/bert-base-uncased", "phishing": "google-bert/bert-base-uncased",
         "fake_news": "google-bert/bert-base-uncased", "product_reviews": "google-bert/bert-base-uncased",
         "political_statements": "google-bert/bert-base-uncased",
         "twitter_rumours": "FacebookAI/roberta-base", "job_scams": "FacebookAI/roberta-base"}


def main(pred_dir, tfidf_path, out="summary.png"):
    runs = load(pred_dir)
    tf = json.loads(open(tfidf_path).read())
    doms = list(FINAL)
    series = {"All-deceptive (trivial)": [], "TF-IDF + LogReg": [], "Fine-tuned transformer (ours)": [],
              "Paper fine-tune": []}
    for d in doms:
        rs = runs[(d, FINAL[d])]
        y = np.array(rs[0]["test"]["y"])
        p = np.mean([r["test"]["p1"] for r in rs], axis=0)  # mean of seed probabilities
        series["All-deceptive (trivial)"].append(f1_score(y, np.ones_like(y), zero_division=0))
        series["TF-IDF + LogReg"].append(tf[d]["f1_05"])
        series["Fine-tuned transformer (ours)"].append(f1_score(y, (p >= 0.5).astype(int), zero_division=0))
        series["Paper fine-tune"].append(PAPER_FT[d])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w, x = 0.2, np.arange(len(doms))
    colors = ["#b0b0b0", "#e8a33d", "#2b6cb0", "#2f855a"]
    for i, (k, v) in enumerate(series.items()):
        ax.bar(x + (i - 1.5) * w, v, w, label=k, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace("_", "\n") for d in doms], fontsize=9)
    ax.set_ylabel("Test F1 (deceptive = positive, threshold 0.5)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.set_title("DIFrauD test F1: simple baselines vs fine-tuned transformers")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
