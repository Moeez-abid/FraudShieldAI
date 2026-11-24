"""Fine-tune DistilBERT for 3-class message classification (ham / spam / scam).

* Class-weighted cross-entropy (scam and spam are minority classes).
* Early stopping on validation macro-F1; the best checkpoint is kept.
* Optional small sweep over learning rate x max_length (--sweep).
* Metrics use the same `message_metrics` as the baseline, so results are comparable.
Test metrics are only computed with --test.

Runs on CUDA (Colab T4), Apple MPS or CPU. Needs requirements-dl.txt.
"""

from __future__ import annotations

import argparse
import itertools
import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from fraudshield.evaluation.metrics import message_metrics, plot_confusion
from fraudshield.utils import git_commit, load_config, resolve, timestamp, write_json


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = len(y) / (n_classes * np.maximum(counts, 1))  # same formula as sklearn "balanced"
    return torch.tensor(w, dtype=torch.float)


def load_split(cfg: dict, name: str) -> pd.DataFrame:
    return pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / f"messages_{name}.parquet")


def train_one(cfg: dict, lr: float, max_length: int, out_dir, train, val, label2id) -> tuple:
    d = cfg["distilbert"]
    set_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(d["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        d["base_model"], num_labels=len(label2id),
        id2label={i: l for l, i in label2id.items()}, label2id=label2id,
    )
    ytr = train.label.map(label2id).to_numpy()
    yva = val.label.map(label2id).to_numpy()
    ds_tr = TextDataset(train.text, ytr, tok, max_length)
    ds_va = TextDataset(val.text, yva, tok, max_length)
    labels = list(label2id)

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        pred = logits.argmax(-1)
        return {"macro_f1": message_metrics([labels[i] for i in y], [labels[i] for i in pred], labels)["macro_f1"]}

    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=lr,
        per_device_train_batch_size=d["batch_size"],
        gradient_accumulation_steps=d.get("grad_accum", 1),
        per_device_eval_batch_size=d["batch_size"] * 2,
        num_train_epochs=d["max_epochs"],
        weight_decay=d["weight_decay"],
        warmup_ratio=d["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=cfg["seed"],
        fp16=torch.cuda.is_available() and not d.get("cpu", False),
        use_cpu=d.get("cpu", False),
        dataloader_pin_memory=False,
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=d["patience"])],
        class_weights=class_weights(ytr, len(label2id)),
    )
    trainer.train()
    return trainer, tok


def predict(trainer: Trainer, tok, texts, max_length: int, labels: list[str]) -> list[str]:
    ds = TextDataset(texts, [0] * len(texts), tok, max_length)
    logits = trainer.predict(ds).predictions
    return [labels[i] for i in logits.argmax(-1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true", help="grid over lr x max_length")
    ap.add_argument("--test", action="store_true", help="also evaluate on the held-out test set")
    ap.add_argument("--batch-size", type=int, help="per-device batch (low-memory machines: 8)")
    ap.add_argument("--grad-accum", type=int, help="gradient accumulation steps (e.g. 2 with batch 8)")
    ap.add_argument("--cpu", action="store_true", help="force CPU (e.g. when MPS runs out of memory)")
    args = ap.parse_args()

    cfg = load_config("message")
    d = cfg["distilbert"]
    if args.batch_size:
        d["batch_size"] = args.batch_size
    if args.grad_accum:
        d["grad_accum"] = args.grad_accum
    d["cpu"] = args.cpu
    labels = cfg["labels"]
    label2id = {l: i for i, l in enumerate(labels)}
    train, val = load_split(cfg, "train"), load_split(cfg, "val")
    work = resolve(cfg["paths"]["model_dir"]) / "distilbert_runs"

    grid = list(itertools.product(d["lr_grid"], d["max_length_grid"])) if args.sweep \
        else [(d["lr"], d["max_length"])]
    best = None
    search = {}
    for lr, max_len in grid:
        run_dir = work / f"lr{lr}_len{max_len}"
        trainer, tok = train_one(cfg, lr, max_len, run_dir, train, val, label2id)
        f1 = message_metrics(val.label, predict(trainer, tok, val.text, max_len, labels), labels)["macro_f1"]
        search[f"lr={lr},max_length={max_len}"] = f1
        print(f"[distilbert] lr={lr} max_length={max_len}: val macro-F1={f1:.4f}", flush=True)
        if best is None or f1 > best[0]:
            best = (f1, lr, max_len, trainer, tok)

    _, lr, max_len, trainer, tok = best
    results = {
        "model": "distilbert",
        "base_model": d["base_model"],
        "best": {"lr": lr, "max_length": max_len},
        "search_val_macro_f1": search,
        "val": message_metrics(val.label, predict(trainer, tok, val.text, max_len, labels), labels),
        "epochs_trained": float(trainer.state.epoch),
        "device": str(trainer.args.device),
        "effective_batch": d["batch_size"] * d.get("grad_accum", 1),
    }
    if args.test:
        test = load_split(cfg, "test")
        results["test"] = message_metrics(test.label, predict(trainer, tok, test.text, max_len, labels), labels)

    final_dir = resolve(cfg["paths"]["model_dir"]) / "distilbert"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    write_json({"max_length": max_len, "labels": labels}, final_dir / "fraudshield.json")
    shutil.rmtree(work, ignore_errors=True)  # checkpoints are large; keep only the best model

    stamp = timestamp()
    results.update({"timestamp": stamp, "git_commit": git_commit(), "train_rows": len(train),
                    "seed": cfg["seed"]})
    write_json(results, resolve("reports/metrics") / f"distilbert_{stamp}.json")
    for split in [s for s in ("val", "test") if s in results]:
        cm = results[split]["confusion_matrix"]
        plot_confusion(cm["matrix"], cm["labels"], f"DistilBERT ({split})",
                       resolve("reports/figures") / f"cm_distilbert_{split}.png")

    v = results["val"]
    print(f"[distilbert] best lr={lr} max_length={max_len} | val macro-F1={v['macro_f1']:.4f} "
          f"| ham FPR={v['fpr_benign_flagged']:.4f}")
    for lab, pc in v["per_class"].items():
        print(f"  {lab:5s} P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} n={pc['support']}")
    for name, sl in v["slices"].items():
        print(f"  {name}: flagged recall={sl['recall_flagged']:.3f} exact-class recall={sl['recall_exact_class']:.3f}")


if __name__ == "__main__":
    main()
