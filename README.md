# Spotting deception in text: a DIFrauD project

This project trains simple, well-known language models (BERT and RoBERTa) to tell **deceptive** texts from **genuine** ones, using the DIFrauD dataset released by Prof. Rakesh Verma's group at the University of Houston. I then compare my scores with the ones published in the papers, and I try to be honest about where I match them, where I don't, and what I am not sure about.

## The short version

- **The data.** DIFrauD has about 96,000 short texts from 7 different areas ("domains"): SMS messages, phishing emails, fake news, product reviews, Twitter rumours, political statements and job-scam ads. Each text is labeled deceptive (1) or genuine (0). The data is already split into 80% training, 10% validation and 10% test.
- **What I did.** For each domain I fine-tuned a pre-trained language model on the training part, picked settings using the validation part, and scored it once on the test part.
- **How it went.** Averaged over the 7 domains, my score is 0.8391. The paper's plain fine-tuning baseline is 0.8399, so I am level with it, not better. I match or beat it on 5 of the 7 domains. I did not reimplement the paper's more advanced methods, and those score a bit higher (0.8635).
- **A simple baseline helps.** I also built a much simpler model (TF-IDF with logistic regression) to see how much the big models really add. On some domains they add a lot, on others almost nothing. Details below.
- **Honest limits.** Some of my results rest on one run, a few domains are close to chance, and I list every check that I did and did not do.

## Words used on this page

- **Domain**: one of the 7 kinds of text (SMS, phishing, and so on). Each is its own yes/no problem.
- **F1**: a score from 0 to 1 that balances two mistakes: calling a genuine text deceptive, and missing a deceptive one. Higher is better. Here "deceptive" is the class we are looking for.
- **Trivial F1**: the F1 you would get by simply answering "deceptive" for every text, without looking at it. It is a floor. If a model barely beats it, it has learned little.
- **AUC**: how well the model ranks deceptive texts above genuine ones. 1.0 is perfect, 0.5 is a coin flip. It does not depend on where you draw the line between "deceptive" and "genuine".
- **Threshold**: the probability above which the model says "deceptive". I use 0.5 unless I say otherwise.
- **Confidence interval (CI)**: a range that shows how much the score could move just because the test set is a finite sample. A wide range means the number is shaky.
- **Seed / ensemble**: training has randomness. A seed fixes it. An ensemble averages the answers of several runs with different seeds, which is steadier than a single run.

## Results

Test-set F1, with "deceptive" as the class we look for. The CI is a 95% bootstrap interval on the test set. It covers test-sampling noise only, not training randomness.

| Domain | Model | Runs | Test F1 @0.5 [95% CI] | Trivial F1 | AUC | AP | Paper fine-tune |
|---|---|---|---|---|---|---|---|
| SMS | BERT-base | 1 | 0.9883 [0.974, 1.000] | 0.326 | 1.000 | 0.999 | 0.9883 |
| Phishing | BERT-base | 1 | 0.9860 [0.979, 0.992] | 0.569 | 0.999 | 0.999 | 0.9859 |
| Fake news | BERT-base | 1 | 0.9841 [0.978, 0.990] | 0.603 | 0.997 | 0.998 | 0.9773 |
| Product reviews | BERT-base | 1 | 0.7192 [0.698, 0.740] | 0.667 | 0.784 | 0.795 | 0.6968 |
| Twitter rumours | RoBERTa-base | 3 (ensemble) | 0.8199 [0.775, 0.860] | 0.508 | 0.939 | 0.898 | 0.7944 |
| Political statements | BERT-base | 3 (ensemble) | 0.7427 [0.718, 0.765] | 0.783 | 0.673 | 0.773 | 0.7640 |
| Job scams | RoBERTa-base | 3 (ensemble) | 0.6337 [0.511, 0.738] | 0.081 | 0.962 | 0.725 | 0.6729 |

(AP is average precision, another threshold-free score. The last column is the plain fine-tuning row of the published table.)

Single-run rows use seed 42. "Ensemble" rows average the predicted probabilities of seeds 42, 1 and 2. For the three-seed domains, the individual runs average (mean ± std) Twitter 0.8205 ± 0.008, Political 0.7326 ± 0.013 and Job scams 0.6329 ± 0.060. The full table is in `results_pred/analysis.md`.

**What the table says in plain words**

- On SMS, phishing and fake news the task is easy and the models are near perfect.
- I match or beat the paper's plain fine-tuning on 5 of 7 domains. On Twitter, Political and Job scams the paper's number falls inside my confidence interval, so I cannot claim a real difference either way.
- **Political statements** is hard for text-only models (AUC about 0.67). The trivial answer "everything is deceptive" scores 0.783, which is higher than my model and higher than the paper's plain fine-tune (0.764). So F1 on the deceptive class says little in this domain.
- **Job scams** is ranked well (AUC 0.96) but F1 is held back because the test set has only 60 deceptive texts, and the model is confidently wrong on some of them (see Error analysis).

## How this compares with the other published results

The paper "LLMs for Explainable Few-shot Deception Detection" (IWSPA 2024, Boumber, Tuck, Verma, Qachfar) uses the same splits and repeats the baseline numbers of the LREC-COLING 2024 paper, so one table covers both. Rows below are copied from its Table 2. "Mean" is the plain average over the 7 domains.

| Method | Pols | News | Jobs | Prod | Phish | Twitter | SMS | Mean |
|---|---|---|---|---|---|---|---|---|
| This work (BERT / RoBERTa fine-tuned) | 0.7427 | 0.9841 | 0.6337 | 0.7192 | 0.9860 | 0.8199 | 0.9883 | 0.8391 |
| Paper: fine-tuned bert-base-uncased | 0.7640 | 0.9773 | 0.6729 | 0.6968 | 0.9859 | 0.7944 | 0.9883 | 0.8399 |
| Paper: RADDICL-3 (4-shot LLM) | 0.7786 | 0.8867 | 0.6606 | 0.4058 | 0.9769 | 0.8511 | 0.7475 | 0.7582 |
| Paper: PreXIA + BERT (adapter) | 0.7881 | 0.9881 | 0.7350 | 0.7206 | 0.9896 | 0.8511 | 0.9723 | 0.8635 |

- My mean (0.8391) is level with plain fine-tuning (0.8399). I am higher on fake news, product reviews and Twitter, level on SMS and phishing, and lower on Political and Job scams.
- I am above the few-shot LLM method (RADDICL-3, mean 0.7582) on 4 of 7 domains (fake news, product reviews, phishing, SMS). It is higher on Political, Job scams and Twitter.
- I am below the adapter method PreXIA (mean 0.8635). I did not reimplement the adapter methods (PreXIA, MAM, UniPELT) or the LLM system, so I cannot claim to beat them.
- On Political statements, only some adapter rows are above the all-"deceptive" answer (0.783), and only narrowly (MAM 0.7841, PreXIA 0.7881).
- About the counts: the paper's text gives 599 deceptive job scams (60 in a 10% test split), which matches my test set. For phishing the IWSPA text mentions 14,219 texts (about 1,422 in the test split), but the dataset page lists 15,272 texts with a test split of 1,528, which is the size I used. So that difference is between the paper's text and the released data.
- A second IWSPA 2024 paper (Hanks et al., about LIWC features) uses different data, a Random Forest, 5-fold cross-validation and a different score, so its numbers cannot be compared with these and I did not reproduce it.

## A simple baseline: is the big model really needed?

I also trained TF-IDF + logistic regression. This is a classic method that just counts which words and word pairs appear, with no pre-trained language model. It runs on a CPU in seconds. I fixed its settings (C=4) before looking at results, so nothing was tuned on the test set. One run per domain, no CI. Run it with `python -m src.tfidf`.

| Domain | TF-IDF F1 @0.5 | TF-IDF F1 @val-tuned thr | AUC | AP | Transformer F1 @0.5 |
|---|---|---|---|---|---|
| SMS | 0.9606 | 0.9572 | 0.998 | 0.993 | 0.9883 |
| Phishing | 0.9758 | 0.9792 | 0.999 | 0.998 | 0.9860 |
| Fake news | 0.9338 | 0.9324 | 0.986 | 0.980 | 0.9841 |
| Product reviews | 0.6526 | 0.6966 | 0.720 | 0.722 | 0.7192 |
| Twitter rumours | 0.8043 | 0.8116 | 0.940 | 0.897 | 0.8199 |
| Political statements | 0.7523 | 0.7838 | 0.650 | 0.761 | 0.7427 |
| Job scams | 0.4810 | 0.6292 | 0.952 | 0.729 | 0.6337 |

To know if a gap is real or just luck, I ran a **paired bootstrap** (`python -m src.paired results_pred`): resample the same test texts 2000 times and see whether "transformer minus TF-IDF" stays on one side of zero. The table shows F1(transformer) minus F1(TF-IDF) at threshold 0.5.

| Domain | Difference | 95% CI | Real difference? |
|---|---|---|---|
| SMS | +0.028 | [+0.006, +0.052] | yes, transformer better |
| Phishing | +0.010 | [+0.001, +0.020] | borderline, transformer slightly better |
| Fake news | +0.050 | [+0.039, +0.062] | yes, transformer better |
| Product reviews | +0.067 | [+0.045, +0.090] | yes, transformer better |
| Political statements | -0.010 | [-0.030, +0.011] | no |
| Twitter rumours | +0.016 | [-0.026, +0.058] | no |
| Job scams | +0.153 | [+0.038, +0.285] | yes at 0.5, but see below |

- **Job scams.** The big gap at 0.5 mostly comes from TF-IDF's probabilities being badly scaled, not from it understanding less. When each model's threshold is chosen on the validation set, both land at about the same F1 (TF-IDF 0.629, transformer 0.628), and their AP is 0.729 versus 0.725. They rank texts about equally well.
- The intervals cover test-sampling noise only. The transformer rows for Political, Twitter and Job scams are 3-seed ensembles, while TF-IDF is a single model.
- On Product reviews, TF-IDF at 0.5 (0.653) is below the trivial answer (0.667).
- The words with the biggest weights hint at shortcuts (hypotheses, not proof). In Twitter they are event names such as charliehebdo and ferguson. In phishing the strongest "genuine" words include enron, dnc, 2001 and 2002, so genuine and phishing emails may come from different sources. In Political they include obamacare, as in the transformer's errors.

![summary](summary.png)

## What I tried to improve things

I chose models only from the validation set, and scored the test set once per run. Numbers are validation F1.

| Setup | Job scams | Political | Twitter |
|---|---|---|---|
| BERT-base, 256 tokens, 3 epochs (default) | 0.7395 | 0.7393 | 0.8208 |
| BERT-base, 512 tokens, 5 epochs (paper's length and epochs) | 0.7395 | 0.7104 | 0.8309 |
| BERT-base, class-weighted loss | 0.7328 | 0.6671 | 0.7991 |
| RoBERTa-base | 0.7500 | 0.7206 | 0.8298 |
| BERT-base, learning rate 5e-5 | n/a | 0.7083 | 0.8284 |

Final choice: BERT-base with default settings for fake news, phishing, product reviews, SMS and political statements, and RoBERTa-base with default settings for Job scams and Twitter. Settings: learning rate 2e-5, 3 epochs, batch size 32, max length 256, fp16 (half-precision training).

**What did not work**

- *Class-weighted loss* (making mistakes on the rare class cost more) lowered validation F1 on all three domains and made training unstable (huge gradients on Job scams).
- *Undersampling* (throwing away most genuine texts to balance the classes) on Job scams left 958 training texts and gave validation F1 of 0.25 to 0.39, far below normal training (0.74 to 0.76). Tuning the threshold raised it to 0.49 to 0.65, still well below. The training set shrank by about 90%.
- *Tuning the threshold on validation* did not help on Job scams or Twitter. On Political statements it raised F1 from 0.743 to 0.783, but at threshold 0.17 the model calls almost everything deceptive (407 of 445 genuine texts wrongly flagged). That is exactly the trivial answer's 0.783, so it is not a real improvement.

## Where the model goes wrong (hypotheses, not verified)

- **Job scams.** At threshold 0.5, 29 of the 60 deceptive test texts were missed, often with probability near 0, and many of them read like ordinary job ads. Some texts labeled genuine contain classic scam cues. This points to label noise or to clues that are not in the text. I did not verify how the labels were made.
- **Political statements.** 11 of the 15 most confident false alarms mention Obama or Obamacare, which looks like a topic shortcut. The missed deceptive statements are mostly neutral statistics.
- **Product reviews.** The IWSPA 2024 LLM paper (Section 5.3) suspects the models pick up the writing pattern of the crowd workers who wrote part of the "deceptive" reviews, rather than true opinion spam. I did not test this.

## Checks on the data

- **Is it the official dataset?** On 4 Nov 2026 I compared all 21 `.jsonl` files of the official repo `redasers/difraud` (the ReDAS Lab at the University of Houston, named on the dataset page) with the files used here from `difraud/difraud`. They are byte-for-byte identical (same SHA-256).
- Job scams: no text appears in both train and test.
- Phishing (`python -m src.overlap --domain phishing`, which prints counts only):
  - 14 of 1,528 test texts are identical (exact match on the `text` field) to a train text, and all 14 have the same label as that train text. Removing them changes test F1 from 0.9860 to 0.9857.
  - Train contains 79 extra copies of texts already in train, and none of these groups has conflicting labels.
  - If matching ignores capital letters, spacing and links, the counts become 20 test texts in train and 101 extra train copies, again with no label conflicts.
  - The dataset page says duplicates within a domain were dropped. A few identical texts remain by these checks. I left the test split unchanged so scores stay comparable with the paper.

## Things to keep in mind

- Training on a GPU is not bit-for-bit repeatable, even with a fixed seed. For example, Job scams (RoBERTa, seed 42) gave test F1 0.6465 and 0.6139 on two runs. Treat single-run numbers as approximate.
- Seeds 42, 1 and 2 were used only for Twitter, Political and Job scams. The other four domains have one run each.
- The confidence intervals do not include training randomness or the choice of threshold.
- Only BERT-base and RoBERTa-base were tested. The paper's adapter methods (including PreXIA) were not reimplemented.

## How to run it

Tested on Google Colab (T4 GPU, Python 3.13).

```
pip install -r requirements.txt
# data_manifest.json is already in the repo. Do NOT re-pin it, so you use exactly the same dataset version.
python -m src.train --domain sms                                   # quick check, test F1 near 0.99
bash run_all.sh google-bert/bert-base-uncased "42"                 # all 7 domains
python -m src.train --domain job_scams --model FacebookAI/roberta-base --seed 42 --out results_pred
python -m src.analyze results_pred                                 # table with CIs, AUC, trivial F1
python -m src.threshold results_pred                               # validation-chosen thresholds
python -m src.errors --domain job_scams --run results_pred/<file>.json --kind fn --n 25
python -m src.dupes --domain phishing --run results_pred/<file>.json
python -m src.overlap --domain phishing                            # repeated-text check
python -m pytest -q tests                                          # unit tests
```

`src/data.py pin` creates `data_manifest.json`. It holds the dataset commit (`aaaf94b336c563a14806bb4f3f58727bed9ed8d4`) and a SHA-256 fingerprint for each file, so the code can refuse any file that has changed.

## Safety: why this project is safe to run

The dataset contains real scam and phishing texts, so I treated it as untrusted from the start.

- The texts are only read as plain text (`json.loads`) and cut into word pieces. They are never executed, opened or shown as web pages. Any example printed has its links disguised (`hxxp`). Do not click or open anything in them.
- Only the plain `.jsonl` data files are downloaded. The dataset's own loading script is never run.
- Hugging Face's scanner marks three files in the dataset repo (the three files in the `phishing/` folder; the other six domains are not flagged) as unsafe. That fits what they are: real phishing emails, which can trigger a link-and-script pattern match. They are only parsed as text on a temporary Colab machine, never run, and no dataset file is stored in this repository. Please do not download the data files to a personal computer or open them in a browser or editor.
- The dataset version is pinned by commit, and every file is checked against its SHA-256 fingerprint before use.
- Models are loaded only in the `safetensors` format (which cannot hide code, unlike pickle), `trust_remote_code` is off, and only two models are allowed.
- No model checkpoints are saved. Saved prediction files hold labels and scores only, never any text.
- No passwords, tokens or keys are used. Data, model files and keys are excluded by `.gitignore`.

## What is in the repo

```
src/            data.py, train.py, analyze.py, threshold.py, errors.py, dupes.py, overlap.py,
                aggregate.py, tfidf.py, paired.py, figure.py, transfer.py
tests/          unit tests (pytest): data checks, overlap check, transfer scoring
results*/       one JSON per run (metrics, settings, library versions, saved scores)
results_pred/   final runs with saved predictions, plus analysis.md
results_tfidf/  TF-IDF baseline results (tfidf.json, tfidf.md)
data_manifest.json   pinned dataset version and file fingerprints
run_all.sh      runs all 7 domains
run_transfer.sh optional cross-domain experiment (code only; no results reported here)
```
