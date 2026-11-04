#!/bin/bash
# Usage: bash run_all.sh MODEL "SEEDS" [extra train.py flags]
# Example: bash run_all.sh google-bert/bert-base-uncased "42" 
#          bash run_all.sh FacebookAI/roberta-base "42 1 2" --balance
set -euo pipefail
MODEL=${1:?model}; SEEDS=${2:-42}; shift 2 || shift 1
for d in fake_news job_scams phishing political_statements product_reviews sms twitter_rumours; do
  for s in $SEEDS; do
    python -m src.train --domain "$d" --model "$MODEL" --seed "$s" "$@"
  done
done
python -m src.aggregate
