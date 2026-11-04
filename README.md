# DIFrauD deception detection

Code and results for detecting deceptive text on the DIFrauD dataset (Boumber, Qachfar, Verma, LREC-COLING 2024) from the ReDAS Lab at the University of Houston.

For each of the 7 domains, I fine-tune a BERT or RoBERTa model to tell deceptive texts from genuine ones. I report test F1 (deceptive = positive class) on the released 80/10/10 splits and compare it with the published results.

## Dataset

DIFrauD has about 96,000 English texts in 7 domains: SMS, phishing, fake news, product reviews, Twitter rumours, political statements and job scams. Each text has a binary label (1 = deceptive, 0 = genuine). The train, validation and test splits are the ones released with the dataset (seed 42).

## Terms

- **F1**: a score from 0 to 1 that balances false alarms and missed deceptive texts. Higher is better.
- **Trivial F1**: the F1 of a classifier that answers "deceptive" for every text. It shows how little a model has learned if it scores close to it.
- **AUC / AP**: scores that measure how well the model ranks deceptive texts above genuine ones. They do not depend on a threshold. AUC of 0.5 is chance.
- **Threshold**: the probability above which the model answers "deceptive". It is 0.5 unless stated.
- **CI**: 95% bootstrap confidence interval on the test set. It covers test-sampling noise only, not training randomness.
- **Ensemble**: the average of the predicted probabilities of several runs with different seeds.

## Results

| Domain | Model | Runs | Test F1 @0.5 [95% CI] | Trivial F1 | AUC | AP | Paper fine-tune |
|---|---|---|---|---|---|---|---|
| SMS | BERT-base | 1 | 0.9883 [0.974, 1.000] | 0.326 | 1.000 | 0.999 | 0.9883 |
| Phishing | BERT-base | 1 | 0.9860 [0.979, 0.992] | 0.569 | 0.999 | 0.999 | 0.9859 |
| Fake news | BERT-base | 1 | 0.9841 [0.978, 0.990] | 0.603 | 0.997 | 0.998 | 0.9773 |
| Product reviews | BERT-base | 1 | 0.7192 [0.698, 0.740] | 0.667 | 0.784 | 0.795 | 0.6968 |
| Twitter rumours | RoBERTa-base | 3 (ensemble) | 0.8199 [0.775, 0.860] | 0.508 | 0.939 | 0.898 | 0.7944 |
| Political statements | BERT-base | 3 (ensemble) | 0.7427 [0.718, 0.765] | 0.783 | 0.673 | 0.773 | 0.7640 |
| Job scams | RoBERTa-base | 3 (ensemble) | 0.6337 [0.511, 0.738] | 0.081 | 0.962 | 0.725 | 0.6729 |

Single-run rows use seed 42. Ensemble rows average seeds 42, 1 and 2. The individual runs for these three domains average (mean ± std): Twitter 0.8205 ± 0.008, Political 0.7326 ± 0.013, Job scams 0.6329 ± 0.060. The full table is in `results_pred/analysis.md`.

Summary:
- The mean F1 over the 7 domains is 0.8391, level with the paper's plain fine-tuning (0.8399).
- F1 is equal or higher than the paper's plain fine-tuning on 5 of 7 domains. On Twitter, Political and Job scams the paper's number is inside my confidence interval, so no difference can be claimed.
- Political statements is close to chance for text-only models (AUC about 0.67). The trivial classifier scores 0.783, higher than my result and higher than the paper's 0.764, so F1 is a weak measure for this domain.
- Job scams ranks well (AUC 0.96), but F1 is limited because the test set has only 60 deceptive texts and some are missed with high confidence (see Error analysis).

## Comparison with published results

The IWSPA 2024 paper "LLMs for Explainable Few-shot Deception Detection" (Boumber, Tuck, Verma, Qachfar) uses the same splits and repeats the baselines of the COLING 2024 paper. The rows below are copied from its Table 2. "Mean" is the plain average over the 7 domains.

| Method | Pols | News | Jobs | Prod | Phish | Twitter | SMS | Mean |
|---|---|---|---|---|---|---|---|---|
| This work (BERT / RoBERTa fine-tuned) | 0.7427 | 0.9841 | 0.6337 | 0.7192 | 0.9860 | 0.8199 | 0.9883 | 0.8391 |
| Paper: fine-tuned bert-base-uncased | 0.7640 | 0.9773 | 0.6729 | 0.6968 | 0.9859 | 0.7944 | 0.9883 | 0.8399 |
| Paper: RADDICL-3 (4-shot LLM) | 0.7786 | 0.8867 | 0.6606 | 0.4058 | 0.9769 | 0.8511 | 0.7475 | 0.7582 |
| Paper: PreXIA + BERT (adapter) | 0.7881 | 0.9881 | 0.7350 | 0.7206 | 0.9896 | 0.8511 | 0.9723 | 0.8635 |

- Against plain fine-tuning: higher on fake news, product reviews and Twitter, level on SMS and phishing, lower on Political and Job scams.
- Against RADDICL-3: higher on fake news, product reviews, phishing and SMS; lower on Political, Job scams and Twitter.
- Against PreXIA: lower in the mean. The adapter methods (PreXIA, MAM, UniPELT) and the LLM system were not reimplemented here.
- On Political statements, the all-deceptive answer (0.783) is above most rows. Only some adapter rows are above it, narrowly (MAM 0.7841, PreXIA 0.7881).
- Counts: the paper gives 599 deceptive job scams (60 in a 10% test split), which matches this test set. For phishing, the IWSPA text mentions 14,219 texts (about 1,422 in the test split). The dataset page lists 15,272 texts with a test split of 1,528, which is the size used here.
- The other IWSPA 2024 paper (Hanks et al., LIWC features) uses different data, a Random Forest, 5-fold cross-validation and weighted F1. It is not comparable and was not reproduced.

## Baseline: TF-IDF + logistic regression

A simple model that counts words and word pairs, with no pre-trained language model. Settings were fixed before looking at results (C=4) and not tuned on test. One run per domain, no CI. Run with `python -m src.tfidf`.

| Domain | TF-IDF F1 @0.5 | TF-IDF F1 @val-tuned thr | AUC | AP | Transformer F1 @0.5 |
|---|---|---|---|---|---|
| SMS | 0.9606 | 0.9572 | 0.998 | 0.993 | 0.9883 |
| Phishing | 0.9758 | 0.9792 | 0.999 | 0.998 | 0.9860 |
| Fake news | 0.9338 | 0.9324 | 0.986 | 0.980 | 0.9841 |
| Product reviews | 0.6526 | 0.6966 | 0.720 | 0.722 | 0.7192 |
| Twitter rumours | 0.8043 | 0.8116 | 0.940 | 0.897 | 0.8199 |
| Political statements | 0.7523 | 0.7838 | 0.650 | 0.761 | 0.7427 |
| Job scams | 0.4810 | 0.6292 | 0.952 | 0.729 | 0.6337 |

A paired bootstrap on the same test texts (`python -m src.paired results_pred`, 2000 resamples, threshold 0.5) gives the 95% interval of F1(transformer) minus F1(TF-IDF):

| Domain | Difference | 95% CI | Real difference? |
|---|---|---|---|
| SMS | +0.028 | [+0.006, +0.052] | yes, transformer better |
| Phishing | +0.010 | [+0.001, +0.020] | borderline, transformer slightly better |
| Fake news | +0.050 | [+0.039, +0.062] | yes, transformer better |
| Product reviews | +0.067 | [+0.045, +0.090] | yes, transformer better |
| Political statements | -0.010 | [-0.030, +0.011] | no |
| Twitter rumours | +0.016 | [-0.026, +0.058] | no |
| Job scams | +0.153 | [+0.038, +0.285] | yes at 0.5, but see below |

- Job scams: the gap at 0.5 comes mostly from TF-IDF's poorly scaled probabilities. With thresholds chosen on validation, both models reach about the same F1 (TF-IDF 0.629, transformer 0.628), and AP is 0.729 vs 0.725.
- The intervals cover test-sampling noise only. The transformer rows for Political, Twitter and Job scams are 3-seed ensembles; TF-IDF is a single model.
- On Product reviews, TF-IDF at 0.5 (0.653) is below the trivial classifier (0.667).
- The highest-weight words suggest possible shortcuts (hypotheses, not proof): event names in Twitter (charliehebdo, ferguson); enron, dnc, 2001 and 2002 as "genuine" words in phishing, so the two classes may come from different sources; obamacare in Political.

![summary](summary.png)

## What was tried

Models were chosen on validation F1 only. The test split was scored once per run. Numbers are validation F1.

| Setup | Job scams | Political | Twitter |
|---|---|---|---|
| BERT-base, 256 tokens, 3 epochs (default) | 0.7395 | 0.7393 | 0.8208 |
| BERT-base, 512 tokens, 5 epochs (paper's length and epochs) | 0.7395 | 0.7104 | 0.8309 |
| BERT-base, class-weighted loss | 0.7328 | 0.6671 | 0.7991 |
| RoBERTa-base | 0.7500 | 0.7206 | 0.8298 |
| BERT-base, learning rate 5e-5 | n/a | 0.7083 | 0.8284 |

Final choice: BERT-base (default) for fake news, phishing, product reviews, SMS and political statements; RoBERTa-base (default) for Job scams and Twitter. Settings: learning rate 2e-5, 3 epochs, batch size 32, max length 256, fp16.

What did not work:
- Class-weighted loss lowered validation F1 on all three domains and made training unstable (very large gradients on Job scams).
- Undersampling the majority class (Job scams, 958 training texts) gave validation F1 of 0.25 to 0.39, far below plain training (0.74 to 0.76). Tuning the threshold raised it to 0.49 to 0.65, still lower.
- Tuning the threshold on validation did not help on Job scams or Twitter. On Political statements it raised F1 from 0.743 to 0.783, but at threshold 0.17 the model flags almost every text (407 of 445 genuine texts are false positives). That equals the trivial classifier, so it is not a real gain.

## Error analysis

These are hypotheses and were not verified.
- Job scams: at threshold 0.5, 29 of 60 deceptive test texts were missed, often with probability near 0, and many read like normal job ads. Some texts labeled genuine contain classic scam cues. This suggests label noise or cues outside the text.
- Political statements: 11 of the 15 most confident false positives mention Obama or Obamacare, which suggests a topic shortcut. Missed deceptive texts are mostly neutral statistics.
- Product reviews: the IWSPA 2024 LLM paper (Section 5.3) suspects the models pick up the style of the crowd workers who wrote part of the deceptive reviews. This was not tested here.

## Data checks

- Source: on 4 Nov 2026 I compared all 21 `.jsonl` files of the official repo `redasers/difraud` with the files used here from `difraud/difraud`. They are byte-identical (same SHA-256).
- Job scams: no text appears in both train and test.
- Phishing (`python -m src.overlap --domain phishing`, prints counts only):
  - 14 of 1,528 test texts are identical (exact match on `text`) to a train text, all with the same label. Removing them changes test F1 from 0.9860 to 0.9857.
  - Train has 79 extra copies of texts already in train, with no label conflicts.
  - Ignoring case, spacing and links gives 20 test texts in train and 101 extra train copies, again with no label conflicts.
  - The dataset page says duplicates within a domain were dropped, but a few identical texts remain by these checks. The test split was left unchanged so scores stay comparable with the paper.

## Limitations

- GPU training is not bit-for-bit repeatable, even with fixed seeds. Job scams (RoBERTa, seed 42) gave test F1 0.6465 and 0.6139 on two runs.
- Four domains (SMS, phishing, fake news, product reviews) have one run each. Seeds 42, 1 and 2 were used only for Twitter, Political and Job scams.
- The confidence intervals do not include training randomness or threshold selection.
- Only BERT-base and RoBERTa-base were tested.

## Usage

Tested on Google Colab (T4 GPU, Python 3.13).

```
pip install -r requirements.txt
# data_manifest.json is in the repo. Do not re-pin it, so you use the same dataset version.
python -m src.train --domain sms                                   # quick check, test F1 near 0.99
bash run_all.sh google-bert/bert-base-uncased "42"                 # all 7 domains
python -m src.train --domain job_scams --model FacebookAI/roberta-base --seed 42 --out results_pred
python -m src.analyze results_pred                                 # table with CIs, AUC, trivial F1
python -m src.threshold results_pred                               # validation-chosen thresholds
python -m src.errors --domain job_scams --run results_pred/<file>.json --kind fn --n 25
python -m src.dupes --domain phishing --run results_pred/<file>.json
python -m src.overlap --domain phishing                            # repeated-text check
python -m pytest -q tests                                          # unit tests
bash make_report.sh                                                # rebuild all tables and the figure (no GPU needed)
```

`make_report.sh` also saves the library versions it used to `results_pred/versions.txt` (python, torch, transformers, numpy, scikit-learn).

`python -m src.data pin` creates `data_manifest.json`: the dataset commit (`aaaf94b336c563a14806bb4f3f58727bed9ed8d4`) and a SHA-256 for each file.

## Security

The dataset contains real scam and phishing texts, so it is treated as untrusted.

- Texts are only parsed with `json.loads` and tokenized. They are never executed, opened or rendered. Printed examples have links defanged (`hxxp`). Do not click or open anything in them.
- Only the plain `.jsonl` files are downloaded. The dataset repo's loading script is never run.
- Hugging Face's scanner marks three files in the dataset repo (the files in the `phishing/` folder; the other six domains are not flagged) as unsafe. They contain real phishing emails, which can trigger a pattern match on links or scripts. They are only parsed as text on a temporary Colab machine, and no dataset file is stored in this repository. Do not download the data files to a personal computer or open them in a browser or editor.
- The dataset version is pinned by commit, and each file is checked against its SHA-256 before use.
- Models load as `safetensors` only (no pickle), `trust_remote_code` is off, and only two models are allowed.
- No checkpoints are saved. Saved prediction files contain labels and scores only, never texts.
- No tokens or secrets are used. Data, model files and keys are excluded by `.gitignore`.

## Repository layout

```
src/            data.py, train.py, analyze.py, threshold.py, errors.py, dupes.py, overlap.py,
                aggregate.py, tfidf.py, paired.py, figure.py, transfer.py
tests/          unit tests (pytest)
results*/       one JSON per run (metrics, settings, library versions, saved scores)
results_pred/   final runs with saved predictions, plus analysis.md
results_tfidf/  TF-IDF baseline results
data_manifest.json   pinned dataset version and file hashes
run_all.sh      runs all 7 domains
make_report.sh  rebuilds all tables and the figure from saved runs
run_transfer.sh optional cross-domain experiment (code only, no results reported)
```
