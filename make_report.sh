#!/bin/bash
# Rebuild all final tables and the figure from the saved runs in results_pred/.
# Run from the repo folder, after the training runs exist. Needs no GPU.
# Usage: bash make_report.sh
set -euo pipefail

python -m src.analyze results_pred            # main table: F1, CI, AUC, AP, trivial F1
python -m src.threshold results_pred          # thresholds chosen on validation
python -m src.tfidf                           # TF-IDF baseline -> results_tfidf/
python -m src.paired results_pred             # paired bootstrap vs TF-IDF
python -m src.figure results_pred results_tfidf/tfidf.json   # summary.png

# Save the library versions used for this run.
python - <<'PY' > results_pred/versions.txt
import platform, numpy, sklearn, torch, transformers
print("python", platform.python_version())
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("scikit-learn", sklearn.__version__)
PY
echo "Done. Tables are in results_pred/ and results_tfidf/, figure is summary.png."
