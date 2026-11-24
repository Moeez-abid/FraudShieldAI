"""Merge UCI SMS Spam + Mendeley SMS Phishing into one deduplicated table.

Output columns: id, text, label (ham/spam/scam), source.
Deduplication happens here, BEFORE splitting, because the Mendeley set reuses many
UCI messages. Conflicting labels for the same message are resolved as follows:
  * spam vs scam  -> scam (Mendeley's finer-grained relabelling)
  * ham vs spam/scam -> dropped (genuinely ambiguous; counted in the log)
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata

import pandas as pd

from fraudshield.utils import load_config, resolve, write_json

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")


def normalise_text(text: str) -> str:
    """Light normalisation that keeps case and punctuation (urgency cues live there)."""
    text = html.unescape(str(text))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("�", "")
    return _WS.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    """Aggressive key for near-duplicate detection: case, punctuation and digits ignored."""
    key = _DIGITS.sub("0", text.lower())
    key = _NON_ALNUM.sub("", key)
    return hashlib.md5(key.encode()).hexdigest()


def load_uci(raw_dir) -> pd.DataFrame:
    path = raw_dir / "uci_sms" / "SMSSpamCollection"
    rows = [line.rstrip("\n").split("\t", 1) for line in path.open(encoding="utf-8")]
    df = pd.DataFrame(rows, columns=["src_label", "text"])
    df["source"] = "uci_sms"
    return df


def load_mendeley(raw_dir) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "mendeley_smishing" / "Dataset_5971.csv", encoding="utf-8")
    df = df.rename(columns={"LABEL": "src_label", "TEXT": "text"})[["src_label", "text"]]
    df["source"] = "mendeley_smishing"
    return df


def build(cfg: dict) -> tuple[pd.DataFrame, dict]:
    raw_dir = resolve(cfg["paths"]["raw_dir"])
    df = pd.concat([load_uci(raw_dir), load_mendeley(raw_dir)], ignore_index=True)
    stats = {"raw_rows": {s: int(n) for s, n in df.source.value_counts().items()}}

    df["src_label"] = df.src_label.str.strip().str.lower()
    df["label"] = [cfg["label_map"][s].get(l) for s, l in zip(df.source, df.src_label)]
    unmapped = df.label.isna()
    stats["unmapped_labels"] = int(unmapped.sum())
    df = df[~unmapped]

    df["text"] = df.text.map(normalise_text)
    df = df[df.text.str.len() > 0]
    df["key"] = df.text.map(dedup_key)

    # Resolve label conflicts across duplicates.
    labels_per_key = df.groupby("key").label.agg(lambda s: frozenset(s))
    ambiguous = labels_per_key[labels_per_key.map(lambda s: "ham" in s and len(s) > 1)].index
    stats["dropped_ham_conflicts"] = int(len(ambiguous))
    df = df[~df.key.isin(ambiguous)]
    to_scam = labels_per_key[labels_per_key.map(lambda s: s == {"spam", "scam"})].index
    stats["spam_to_scam_resolved"] = int(len(to_scam))
    df.loc[df.key.isin(to_scam), "label"] = "scam"

    before = len(df)
    # Prefer the Mendeley row when duplicated so 'source' reflects the finer label.
    df = (
        df.sort_values("source", key=lambda s: s.eq("mendeley_smishing"), ascending=False)
        .drop_duplicates("key")
    )
    stats["duplicates_removed"] = int(before - len(df))

    df = df.sort_values("key").reset_index(drop=True)
    df["id"] = "msg_" + df.key.str[:12]
    out = df[["id", "text", "label", "source"]]
    stats["final_rows"] = int(len(out))
    stats["final_label_counts"] = {k: int(v) for k, v in out.label.value_counts().items()}
    stats["final_by_source_label"] = {
        f"{s}/{l}": int(n) for (s, l), n in out.groupby(["source", "label"]).size().items()
    }
    return out, stats


def main() -> None:
    cfg = load_config("message")
    out, stats = build(cfg)
    path = resolve(cfg["paths"]["interim"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    write_json(stats, path.with_suffix(".stats.json"))
    print(f"[clean_messages] {len(out)} rows -> {path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
