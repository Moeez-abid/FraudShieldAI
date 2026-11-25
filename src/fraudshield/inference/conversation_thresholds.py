"""Tune conversation-mode thresholds on synthetic chats built from the VALIDATION split.

    python -m fraudshield.inference.conversation_thresholds   (or: make conversation-thresholds)

Why separate thresholds: the conversation score is the MAXIMUM over many windows, so a
long benign chat gets many chances to cross a single-message threshold. Thresholds are
set so that at most `medium_max_benign_flagged` / `high_max_benign_flagged` of benign
synthetic chats reach Medium / High (configs/conversation.yaml).

Synthetic chats (validation messages only; the test split is never touched):
  benign    L normal messages
  inserted  L-1 normal messages + one spam/scam message at a random position
  split     normal messages + one spam/scam message cut at a sentence boundary into two
            consecutive turns (a scam spread over messages; this is what windows are for)

Limitation: these are random messages stitched together, not real conversations.

Writes models/conversation_thresholds.json and reports/metrics/conversation_thresholds.json.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from fraudshield.inference.conversation import (
    Turn,
    build_pieces,
    score_texts,
    token_budget,
    word_token_counter,
)
from fraudshield.inference.conversation_rules import conversation_floor
from fraudshield.utils import load_config, resolve, timestamp, write_json

_SENT = re.compile(r"(?<=[.!?])\s+")


def _split_in_two(text: str) -> tuple[str, str] | None:
    parts = _SENT.split(text.strip())
    if len(parts) < 2:
        return None
    mid = len(parts) // 2
    return " ".join(parts[:mid]), " ".join(parts[mid:])


def make_chats(val: pd.DataFrame, cfg: dict) -> list[dict]:
    rng = np.random.default_rng(cfg["seed"])
    ham = val[val.label == "ham"].text.tolist()
    mal = val[val.label != "ham"][["text", "label"]].to_dict("records")
    splittable = [(m, s) for m in mal if (s := _split_in_two(m["text"]))]
    lo, hi = cfg["synthetic"]["length"]
    chats = []

    def normal(n):
        return [ham[i] for i in rng.choice(len(ham), size=n, replace=False)]

    for _ in range(cfg["synthetic"]["n_benign"]):
        chats.append({"kind": "benign", "label": "ham", "msgs": normal(int(rng.integers(lo, hi + 1)))})
    for _ in range(cfg["synthetic"]["n_inserted"]):
        m = mal[int(rng.integers(len(mal)))]
        msgs = normal(int(rng.integers(lo, hi + 1)) - 1)
        msgs.insert(int(rng.integers(len(msgs) + 1)), m["text"])
        chats.append({"kind": "inserted", "label": m["label"], "msgs": msgs})
    for _ in range(cfg["synthetic"]["n_split"]):
        m, (a, b) = splittable[int(rng.integers(len(splittable)))]
        msgs = normal(int(rng.integers(lo, hi + 1)) - 2)
        pos = int(rng.integers(len(msgs) + 1))
        msgs[pos:pos] = [a, b]
        chats.append({"kind": "split", "label": m["label"], "msgs": msgs})
    return chats


def benign_threshold(benign_scores: np.ndarray, max_flagged: float) -> float:
    """Smallest threshold t with mean(benign >= t) <= max_flagged (as in risk.py)."""
    s = np.sort(benign_scores)
    k = int(np.floor(max_flagged * len(s)))
    return float(np.nextafter(s[-k - 1], np.inf)) if k < len(s) else float(s[0])


def rule_rates(chats: list[dict]) -> dict:
    """Share of synthetic chats of each kind on which the cross-message rules fire."""
    rates = {}
    for kind in ("benign", "inserted", "split"):
        hits = [conversation_floor([Turn(i + 1, None, t) for i, t in enumerate(c["msgs"])])[0]
                for c in chats if c["kind"] == kind]
        rates[kind] = {"any": float(np.mean([h is not None for h in hits])),
                       "high": float(np.mean([h == "High" for h in hits]))}
    return rates


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-only", action="store_true",
                    help="only recompute the rule trigger rates in the existing thresholds file")
    args = ap.parse_args()

    cfg = load_config("conversation")
    mcfg = load_config("message")
    pdir, mdir = resolve(mcfg["paths"]["processed_dir"]), resolve(mcfg["paths"]["model_dir"])
    val = pd.read_parquet(pdir / "messages_val.parquet")
    chats = make_chats(val, cfg)

    if args.rules_only:
        out = json.loads((mdir / "conversation_thresholds.json").read_text())
        out["rules"] = rule_rates(chats)
        write_json(out, mdir / "conversation_thresholds.json")
        write_json(out, resolve("reports/metrics") / "conversation_thresholds.json")
        print(f"[conversation] cross-message rules fire on: {out['rules']}")
        return

    from fraudshield.inference.message_model import DistilBertMessageModel

    model = DistilBertMessageModel(mdir / "distilbert")
    count, budget = word_token_counter(model), token_budget(model)
    kmax = max(cfg["max_messages_grid"])
    cache: dict = {}
    all_pieces = []
    for c in chats:
        turns = [Turn(i + 1, None, t) for i, t in enumerate(c["msgs"])]
        c["turns"] = turns
        all_pieces.append(build_pieces(turns, count, budget, max_messages=kmax))
    texts = list(dict.fromkeys(p.text for ps in all_pieces for p in ps))
    print(f"[conversation] {len(chats)} synthetic chats, {len(texts)} unique windows to score...")
    score_texts(model, texts, cache)

    kinds = np.array([c["kind"] for c in chats])
    risk = cfg["risk"]
    grid = {}
    for k in cfg["max_messages_grid"]:
        best, best_lab = [], []
        for ps in all_pieces:
            use = [p for p in ps if p.n_messages <= k]
            s, lab = score_texts(model, [p.text for p in use], cache)
            i = int(np.argmax(s))
            best.append(s[i])
            best_lab.append(lab[i])
        best = np.array(best)
        med = benign_threshold(best[kinds == "benign"], risk["medium_max_benign_flagged"])
        high = max(med, benign_threshold(best[kinds == "benign"], risk["high_max_benign_flagged"]))
        row = {"medium": med, "high": high}
        for kind in ("benign", "inserted", "split"):
            m = kinds == kind
            row[kind] = {"n": int(m.sum()),
                         "flagged_medium": float((best[m] >= med).mean()),
                         "flagged_high": float((best[m] >= high).mean())}
        grid[k] = row
        print(f"  max_messages={k}: medium>={med:.4f} high>={high:.4f} | benign flagged "
              f"{row['benign']['flagged_medium']:.3f}/{row['benign']['flagged_high']:.3f} | "
              f"inserted caught {row['inserted']['flagged_medium']:.3f}/{row['inserted']['flagged_high']:.3f} | "
              f"split caught {row['split']['flagged_medium']:.3f}/{row['split']['flagged_high']:.3f}")

    rules = rule_rates(chats)
    print(f"  cross-message rules fire on: {rules}")

    k = cfg["max_messages"]
    out = {"model": "distilbert", "max_messages": k, "budget_tokens": budget,
           "medium": grid[k]["medium"], "high": grid[k]["high"],
           "targets": risk, "synthetic": cfg["synthetic"], "seed": cfg["seed"],
           "source": "validation split only (synthetic chats)", "grid": grid, "rules": rules,
           "created": timestamp()}
    write_json(out, mdir / "conversation_thresholds.json")
    write_json(out, resolve("reports/metrics") / "conversation_thresholds.json")
    print(f"[conversation] chosen max_messages={k}: medium>={out['medium']:.4f} high>={out['high']:.4f}")


if __name__ == "__main__":
    main()
