"""Merge and clean the URL datasets.

Sources
-------
* PhiUSIIL (2024): 235k URLs. Every benign URL is an https homepage with no path,
  so its scheme/path structure is a label shortcut.
* Hannousse & Yahiouche (2020): 11.4k URLs, balanced. Both classes include deep
  links and http/https, so path and scheme features are informative here.

Only URL strings and labels are used. Both datasets also ship page-content features,
which need the page to be fetched; the app never does that.

`feature_url` is the string the "mixed" feature view uses:
  PhiUSIIL  -> host name only, leading "www." removed (both classes; removes the
               homepage/https/path and www shortcuts)
  Hannousse -> full URL
`url` always keeps the original normalised URL.

Output columns: id, url, feature_url, label (1 = phishing), source, reg_domain.
"""

from __future__ import annotations

import hashlib
import html

import pandas as pd

from fraudshield.features.url_features import _split, registered_domain
from fraudshield.utils import load_config, resolve, write_json

# Label-encoding check: the PhiUSIIL paper reports 134,850 legitimate and 100,945
# phishing URLs. (Checking famous domains does not work: phishing pages hosted on
# docs.google.com, s3.amazonaws.com etc. carry those registered domains.)
EXPECTED_PHIUSIIL = {"benign": 134_850, "phishing": 100_945}
EXPECTED_HANNOUSSE = {"benign": 5_715, "phishing": 5_715}


def _normalise(url: str) -> str:
    # The scrape contains HTML-escaped query strings (&amp;); undo that artefact.
    url = html.unescape(str(url)).strip()
    # Lowercase scheme + host only; paths are case-sensitive.
    if "://" in url:
        scheme, rest = url.split("://", 1)
        host, sep, tail = rest.partition("/")
        return f"{scheme.lower()}://{host.lower()}{sep}{tail}"
    host, sep, tail = url.partition("/")
    return f"{host.lower()}{sep}{tail}"


def _check_counts(name: str, labels: pd.Series, expected: dict) -> None:
    got = {"benign": int((labels == 0).sum()), "phishing": int(labels.sum())}
    if got != expected:
        raise ValueError(f"{name}: unexpected class counts {got}; check the label mapping")


def load_phiusiil(cfg: dict) -> pd.DataFrame:
    raw = resolve(cfg["paths"]["raw_dir"]) / "phiusiil"
    csvs = sorted(raw.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV under {raw}; run `make data` first")
    df = pd.read_csv(csvs[0], usecols=["URL", "label"]).rename(columns={"URL": "url"})
    df["label"] = (df.label == cfg["phiusiil_phishing_value"]).astype(int)
    _check_counts("phiusiil", df.label, EXPECTED_PHIUSIIL)
    df["source"] = "phiusiil"
    return df


def load_hannousse(cfg: dict) -> pd.DataFrame:
    path = resolve(cfg["paths"]["raw_dir"]) / "hannousse" / "dataset_B_05_2020.csv"
    df = pd.read_csv(path, usecols=["url", "status"])
    df["label"] = (df.status.str.strip().str.lower() == "phishing").astype(int)
    _check_counts("hannousse", df.label, EXPECTED_HANNOUSSE)
    df["source"] = "hannousse"
    return df[["url", "label", "source"]]


def build(cfg: dict) -> tuple[pd.DataFrame, dict]:
    df = pd.concat([load_phiusiil(cfg), load_hannousse(cfg)], ignore_index=True)
    stats = {"raw_rows": {s: int(n) for s, n in df.source.value_counts().items()}}

    df = df.dropna(subset=["url"])
    df["url"] = df.url.map(_normalise)
    df = df[df.url.str.len().between(4, 2048)]

    before = len(df)
    conflicts = df.groupby("url").label.nunique()
    conflicting = conflicts[conflicts > 1].index
    df = df[~df.url.isin(conflicting)]
    # Prefer the Hannousse copy of a duplicated URL: its full-URL structure is trustworthy.
    df = (df.sort_values("source", key=lambda s: s.eq("hannousse"), ascending=False)
          .drop_duplicates("url"))
    stats["conflicting_urls_dropped"] = int(len(conflicting))
    stats["duplicates_removed"] = int(before - len(df) - len(conflicting))

    df["reg_domain"] = df.url.map(registered_domain)
    df = df[df.reg_domain.str.len() > 0]
    # PhiUSIIL: host only, and a leading "www." removed for both classes, since 100 % of
    # its benign hosts start with www. (vs 41 % of phishing): another collection artefact.
    phi_host = df.url.map(lambda u: _split(u)[1]).str.replace(r"^www\.", "", regex=True)
    df["feature_url"] = df.url.where(df.source == "hannousse", phi_host)

    df["id"] = "url_" + df.url.map(lambda u: hashlib.md5(u.encode()).hexdigest()[:12])
    out = (df[["id", "url", "feature_url", "label", "source", "reg_domain"]]
           .sort_values("id").reset_index(drop=True))

    stats["final_rows"] = len(out)
    stats["by_source_label"] = {
        f"{s}/{'phishing' if l else 'benign'}": int(n)
        for (s, l), n in out.groupby(["source", "label"]).size().items()
    }
    stats["n_domains"] = int(out.reg_domain.nunique())
    # Structural quirks per source, so artefacts can be discussed in the report.
    no_scheme = out.url.str.replace(r"^[a-z]+://", "", regex=True)
    out_q = out.assign(has_path=no_scheme.str.contains(r"/.", regex=True),
                       https=out.url.str.startswith("https"))
    stats["share_with_path"] = {
        f"{s}/{'phishing' if l else 'benign'}": round(float(v), 3)
        for (s, l), v in out_q.groupby(["source", "label"]).has_path.mean().items()
    }
    stats["share_https"] = {
        f"{s}/{'phishing' if l else 'benign'}": round(float(v), 3)
        for (s, l), v in out_q.groupby(["source", "label"]).https.mean().items()
    }
    return out, stats


def main() -> None:
    cfg = load_config("url")
    out, stats = build(cfg)
    path = resolve(cfg["paths"]["interim"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    write_json(stats, path.with_suffix(".stats.json"))
    print(f"[clean_urls] {len(out)} rows -> {path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
