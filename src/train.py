"""Fine-tune one model on one DIFrauD domain. Reports binary F1 (deceptive = 1).

Model selection rule: hyperparameters are chosen on the validation split only.
The test split is evaluated once, after training ends.
"""
import argparse
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import torch
import transformers
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)

from src.data import DOMAINS, load_domain, load_manifest

# Only these models can be loaded. Add a model here only after checking it ships
# a model.safetensors file on its Hugging Face page.
MAX_CHARS = 20_000
ALLOWED_MODELS = ["google-bert/bert-base-uncased", "FacebookAI/roberta-base"]


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *a, class_weights=None, **k):
        super().__init__(*a, **k)
        self.cw = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        w = self.cw.to(out.logits.device) if self.cw is not None else None
        loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w)
        return (loss, out) if return_outputs else loss


def scores(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    ap.add_argument("--model", default=ALLOWED_MODELS[0], choices=ALLOWED_MODELS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--balance", action="store_true", help="class-weighted loss")
    ap.add_argument("--undersample", action="store_true",
                    help="train on all minority-class texts plus an equal-size random sample of the majority class")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    set_seed(a.seed)
    manifest = load_manifest()
    splits, report = load_domain(a.domain, manifest)
    print("Data report:", json.dumps(report))
    if report["leak_train_test"] or report["leak_train_val"]:
        print("WARNING: identical texts appear in train and eval splits. See report.")

    if a.undersample:
        rng = np.random.default_rng(a.seed)
        tr_texts, tr_labels = splits["train"]
        y = np.array(tr_labels)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        minor, major = (pos, neg) if len(pos) < len(neg) else (neg, pos)
        keep = np.sort(np.concatenate([minor, rng.choice(major, size=len(minor), replace=False)]))
        splits["train"] = ([tr_texts[i] for i in keep], [tr_labels[i] for i in keep])
        print(f"Undersampled train to {len(keep)} texts")

    tok = AutoTokenizer.from_pretrained(a.model)
    data, trunc = {}, {}
    for s, (texts, labels) in splits.items():
        texts = [t[:MAX_CHARS] for t in texts]  # speed only; max_len cuts far earlier
        full = tok(texts, truncation=False)["input_ids"]
        trunc[s] = float(np.mean([len(x) > a.max_len for x in full]))
        enc = tok(texts, truncation=True, max_length=a.max_len)
        data[s] = TextDataset(enc, labels)

    cw = None
    if a.balance:
        cnt = np.bincount(splits["train"][1], minlength=2)
        cw = torch.tensor(len(splits["train"][1]) / (2.0 * cnt), dtype=torch.float)

    # use_safetensors=True refuses pickle-based weight files.
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=2, use_safetensors=True, trust_remote_code=False)
    args = TrainingArguments(
        output_dir="ckpt", num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=64,
        eval_strategy="no", save_strategy="no",  # no checkpoints written to disk
        seed=a.seed, fp16=torch.cuda.is_available(), report_to="none")
    tr = WeightedTrainer(model=model, args=args, train_dataset=data["train"],
                         data_collator=DataCollatorWithPadding(tok), class_weights=cw)
    tr.train()

    res, preds = {}, {}
    for s in ("validation", "test"):
        p = tr.predict(data[s])
        res[s] = scores(p.label_ids, p.predictions.argmax(-1))
        probs = torch.softmax(torch.tensor(p.predictions), dim=-1)[:, 1].numpy()
        # Only labels and scores are saved, never the texts.
        preds[s] = {"y": [int(v) for v in p.label_ids], "p1": [round(float(v), 5) for v in probs]}
    res["predictions"] = preds
    res["meta"] = {
        "domain": a.domain, "model": a.model, "seed": a.seed, "balance": a.balance,
        "undersample": a.undersample, "n_train_used": len(splits["train"][1]), "epochs": a.epochs, "lr": a.lr, "max_len": a.max_len, "batch_size": a.batch_size,
        "frac_truncated": trunc, "data_report": report,
        "dataset_revision": manifest["revision"],
        "model_commit": getattr(model.config, "_commit_hash", None),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda": torch.cuda.is_available(), "python": platform.python_version()}
    Path(a.out).mkdir(exist_ok=True)
    name = f"{a.domain}__{a.model.split('/')[-1]}__s{a.seed}{'__bal' if a.balance else ''}{'__us' if a.undersample else ''}.json"
    Path(a.out, name).write_text(json.dumps(res, indent=2))
    print(f"{a.domain} val F1={res['validation']['f1']:.4f} test F1={res['test']['f1']:.4f}")


if __name__ == "__main__":
    main()
