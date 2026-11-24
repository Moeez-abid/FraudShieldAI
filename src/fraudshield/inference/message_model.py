"""Thin wrappers giving both message models the same interface:

    model.labels                       -> list[str]
    model.predict_proba(list[str])     -> np.ndarray [n, n_labels]
    model.predict_log_proba(list[str]) -> np.ndarray [n, n_labels]  (exact; used by explanations)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class BaselineMessageModel:
    name = "TF-IDF + Logistic Regression"

    def __init__(self, path: Path):
        self.pipe = joblib.load(path)
        self.labels = list(self.pipe.classes_)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipe.predict_proba(texts)

    def predict_log_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipe.predict_log_proba(texts)


class DistilBertMessageModel:
    name = "DistilBERT (fine-tuned)"

    def __init__(self, model_dir: Path, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        meta = json.loads((Path(model_dir) / "fraudshield.json").read_text())
        self.max_length = meta["max_length"]
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        id2label = self.model.config.id2label
        self.labels = [id2label[i] for i in range(len(id2label))]
        self.batch_size = batch_size
        # CPU inference is fast enough for single messages and avoids MPS/CUDA quirks.
        self.device = "cpu"

    def predict_log_proba(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], truncation=True,
                               max_length=self.max_length, padding=True, return_tensors="pt")
                logits = self.model(**enc).logits.double()
                out.append(self.torch.log_softmax(logits, -1).numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.exp(self.predict_log_proba(texts))
