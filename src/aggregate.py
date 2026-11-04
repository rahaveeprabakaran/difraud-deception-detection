"""Combine results/*.json into one table next to the published numbers."""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

# Best binary F1 per domain among the BERT methods, LREC-COLING 2024 (Boumber, Qachfar,
# Verma), Table 2. Copied from the paper's table; re-check against the PDF before you cite.
PAPER = {"political_statements": 0.7881, "fake_news": 0.9881, "job_scams": 0.7667,
         "product_reviews": 0.7215, "phishing": 0.9896, "twitter_rumours": 0.8511,
         "sms": 0.9961}


def main(folder="results"):
    groups = defaultdict(list)
    vals = defaultdict(list)
    for p in sorted(Path(folder).glob("*.json")):
        r = json.loads(p.read_text())
        m = r["meta"]
        tag = "undersample" if m.get("undersample") else ("class-weights" if m["balance"] else "-")
        k = (m["model"], tag, m["domain"])
        groups[k].append(r["test"]["f1"])
        vals[k].append(r["validation"]["f1"])
    lines = ["| Model | Balancing | Domain | Seeds | Val F1 (mean) | Test F1 (mean ± std) | Paper best | Diff |",
             "|---|---|---|---|---|---|---|---|"]
    for (model, bal, dom), v in sorted(groups.items()):
        mean = st.mean(v)
        sd = st.stdev(v) if len(v) > 1 else 0.0
        ref = PAPER.get(dom)
        lines.append(f"| {model} | {bal} | {dom} | {len(v)} | {st.mean(vals[(model, bal, dom)]):.4f} | {mean:.4f} ± {sd:.4f} | "
                     f"{ref} | {mean - ref:+.4f} |")
    out = "\n".join(lines)
    Path(folder, "summary.md").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
