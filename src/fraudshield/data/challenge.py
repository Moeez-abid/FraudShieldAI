"""Challenge-set checks: `python -m fraudshield.data.challenge` (or `make challenge-check`).

Reports, for data/challenge/messages.csv and urls.csv:
  * how many rows each annotator has labelled and how many still lack label_final
  * Cohen's kappa between label_moeez and label_ijtaba (independent labels)
  * the rows where the two annotators disagree (to discuss)
  * overlap with the training data (must be zero): messages by dedup key, URLs by
    exact URL and by registered domain (domain overlap is reported, not an error)
  * invalid label values
Writes reports/metrics/challenge_check.json.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from fraudshield.data.clean_messages import dedup_key
from fraudshield.features.url_features import registered_domain
from fraudshield.inference.extract_urls import refang
from fraudshield.utils import resolve, write_json

VALID = {"messages": {"ham", "spam", "scam"}, "urls": {"benign", "phishing"}}
ANNOTATORS = ("label_moeez", "label_ijtaba")


def _norm(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def agreement(df: pd.DataFrame, kind: str) -> dict:
    a, b = (_norm(df[c]) for c in ANNOTATORS)
    both = a.notna() & b.notna()
    out = {
        "rows": len(df),
        "labelled_by": {c: int(_norm(df[c]).notna().sum()) for c in ANNOTATORS},
        "both_labelled": int(both.sum()),
        "final_labelled": int(_norm(df["label_final"]).notna().sum()),
    }
    invalid = {}
    for c in (*ANNOTATORS, "label_final"):
        bad = _norm(df[c]).dropna()
        bad = bad[~bad.isin(VALID[kind])]
        if len(bad):
            invalid[c] = sorted(bad.unique().tolist())
    out["invalid_labels"] = invalid
    if both.sum() >= 2:
        out["raw_agreement"] = float((a[both] == b[both]).mean())
        out["cohen_kappa"] = float(cohen_kappa_score(a[both], b[both]))
        out["disagreements"] = df.loc[both & (a != b), "id"].tolist()
    if out["final_labelled"]:
        out["final_label_counts"] = _norm(df["label_final"]).value_counts().to_dict()
    return out


def message_overlap(msgs: pd.DataFrame) -> list[str]:
    train_keys = set(pd.read_parquet(resolve("data/interim/messages.parquet")).text.map(dedup_key))
    return msgs.loc[msgs.text.map(dedup_key).isin(train_keys), "id"].tolist()


def url_overlap(urls: pd.DataFrame) -> tuple[list[str], list[str]]:
    data = pd.read_parquet(resolve("data/interim/urls.parquet"), columns=["url", "reg_domain"])
    known_urls = set(data.url.str.lower().str.rstrip("/"))
    known_domains = set(data.reg_domain)
    fixed = urls.url_defanged.map(refang)
    full = fixed.map(lambda u: u if "://" in u else "http://" + u)
    exact = urls.loc[fixed.str.lower().str.rstrip("/").isin(known_urls)
                     | full.str.lower().str.rstrip("/").isin(known_urls), "id"].tolist()
    domain = urls.loc[full.map(registered_domain).isin(known_domains), "id"].tolist()
    return exact, domain


def main() -> None:
    d = resolve("data/challenge")
    msgs = pd.read_csv(d / "messages.csv", dtype=str)
    urls = pd.read_csv(d / "urls.csv", dtype=str)
    report = {"messages": agreement(msgs, "messages"), "urls": agreement(urls, "urls")}

    if len(msgs):
        report["messages"]["overlap_with_training"] = message_overlap(msgs)
    if len(urls):
        exact, domain = url_overlap(urls)
        report["urls"]["exact_overlap_with_training"] = exact
        report["urls"]["domain_seen_in_training"] = domain

    write_json(report, resolve("reports/metrics") / "challenge_check.json")
    for kind, r in report.items():
        print(f"[challenge] {kind}: {r['rows']} rows | labelled {r['labelled_by']} | "
              f"final {r['final_labelled']}")
        if "cohen_kappa" in r:
            print(f"  agreement {r['raw_agreement']:.1%}, Cohen's kappa {r['cohen_kappa']:.3f}, "
                  f"{len(r['disagreements'])} disagreements: {r['disagreements'][:10]}")
        if r["invalid_labels"]:
            print(f"  INVALID labels: {r['invalid_labels']} (allowed: {sorted(VALID[kind])})")
        for key in ("overlap_with_training", "exact_overlap_with_training"):
            if r.get(key):
                print(f"  OVERLAP with training data (remove these): {r[key]}")
        if r.get("domain_seen_in_training"):
            print(f"  note: {len(r['domain_seen_in_training'])} URLs on domains seen in training")


if __name__ == "__main__":
    main()
