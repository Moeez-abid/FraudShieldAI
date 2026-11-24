"""Word-occlusion explanations for any message classifier.

For each word, remove it and re-score the message; a word's importance is how much
its removal lowers the malicious LOG-ODDS, log(p / (1 - p)) with p = P(spam) + P(scam).
Log-odds keep working when the model is saturated (p = 0.999 vs 0.99 is a real change
that a plain probability difference would hide). All variants are scored
in one batch, so the cost is a single batched forward pass of up to N+1 rows.

Takes a `predict_log_proba(list[str]) -> np.ndarray[n, n_classes]` callable (both
message models provide one), so the baseline and DistilBERT get identical
explanations. Log-probabilities matter: in float32, softmax rounds P(ham) to exactly 0
for confident predictions, which would hide every word's effect.

Top words are only returned when the prediction is not benign: words "pushing toward
scam" in a message rated safe would confuse users.

Limitation: leave-one-out can't attribute redundant evidence. A message packed with
scam cues stays confidently scam whichever single word is removed. Then
`distributed` is True and the UI should say the overall pattern, not one word,
drives the verdict.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

import numpy as np
from scipy.special import logsumexp

_TOKEN = re.compile(r"\S+")


def occlusion(
    text: str,
    predict_log_proba: Callable[[list[str]], np.ndarray],
    labels: Sequence[str],
    benign: str = "ham",
    max_words: int = 64,
    top_k: int = 5,
    min_importance: float = 0.25,
) -> dict:
    spans = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
    if not spans:
        return {"base_score": 0.0, "words": []}
    spans_used = spans[:max_words]  # cap cost; words beyond the cap get no score

    variants = [text] + [text[:s] + text[e:] for s, e in spans_used]
    logp = np.asarray(predict_log_proba(variants), dtype=float)
    benign_idx = list(labels).index(benign)
    mal_idx = [i for i in range(len(labels)) if i != benign_idx]
    # log-odds of malicious = log(sum P(malicious classes)) - log P(benign), computed stably
    log_odds = logsumexp(logp[:, mal_idx], axis=1) - logp[:, benign_idx]
    probs = np.exp(logp[0])
    base = float(1.0 - probs[benign_idx])
    deltas = log_odds[0] - log_odds[1:]  # positive = word pushed toward malicious

    words = [
        {"word": text[s:e], "start": s, "end": e, "importance": float(d)}
        for (s, e), d in zip(spans_used, deltas)
    ]
    predicted = labels[int(probs.argmax())]
    ranked = sorted(words, key=lambda w: w["importance"], reverse=True)
    top = [] if predicted == benign else \
        [w for w in ranked[:top_k] if w["importance"] > min_importance]
    return {
        "base_score": base,
        "base_log_odds": float(log_odds[0]),
        "predicted": predicted,
        "probs": {l: float(p) for l, p in zip(labels, probs)},
        "top_words": top,
        "distributed": predicted != benign and not top,
        "words": words,
        "truncated": len(spans) > max_words,
    }
