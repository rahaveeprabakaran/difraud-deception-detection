#!/usr/bin/env bash
# Train one BERT-base model per source domain (fixed settings) and score the other domains' test sets.
# Smallest domains first, so you can stop at any point and still analyze what finished.
set -e
for d in sms twitter_rumours job_scams phishing political_statements fake_news product_reviews; do
  python -m src.train --domain "$d" --seed 42 --transfer --out results_transfer
done
python -m src.transfer results_transfer
