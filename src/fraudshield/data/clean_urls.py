"""Clean the PhiUSIIL URL dataset.

Only the raw URL string and label are used. PhiUSIIL's precomputed page-content
features (title match, number of images, etc.) need the page to be fetched, which
the app never does, so using them would make offline results misleading.

Output columns: id, url, label (1 = phishing, 0 = benign), reg_domain.
"""

from __future__ import annotations

import hashlib
import html

import pandas as pd

from fraudshield.features.url_features import registered_domain
from fraudshield.utils import load_config, resolve, write_json

# Label-encoding check: the PhiUSIIL paper reports 134,850 legitimate and 100,945
# phishing URLs. (Checking famous domains does not work: phishing pages hosted on
# docs.google.com, s3.amazonaws.com etc. carry those registered domains.)
EXPECTED_RAW_COUNTS = {"benign": 134_850, "phishing": 100_945}


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


def build(cfg: dict) -> tuple[pd.DataFrame, dict]:
    raw_dir = resolve(cfg["paths"]["raw_dir"]) / "phiusiil"
    csvs = sorted(raw_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV under {raw_dir}; run `make data` first")
    df = pd.read_csv(csvs[0], usecols=["URL", "label"])
    stats = {"source_file": csvs[0].name, "raw_rows": len(df)}

    df["label"] = (df.label == cfg["phiusiil_phishing_value"]).astype(int)
    raw_counts = {"benign": int((df.label == 0).sum()), "phishing": int(df.label.sum())}
    if raw_counts != EXPECTED_RAW_COUNTS:
        raise ValueError(f"Unexpected class counts {raw_counts}; check phiusiil_phishing_value")

    df = df.dropna(subset=["URL"])
    df["url"] = df.URL.map(_normalise)
    df = df[df.url.str.len().between(4, 2048)]

    before = len(df)
    conflicts = df.groupby("url").label.nunique()
    conflicting = conflicts[conflicts > 1].index
    df = df[~df.url.isin(conflicting)].drop_duplicates("url")
    stats["conflicting_urls_dropped"] = int(len(conflicting))
    stats["duplicates_removed"] = int(before - len(df) - len(conflicting))

    df["reg_domain"] = df.url.map(registered_domain)
    df = df[df.reg_domain.str.len() > 0]

    df["id"] = "url_" + df.url.map(lambda u: hashlib.md5(u.encode()).hexdigest()[:12])
    out = df[["id", "url", "label", "reg_domain"]].sort_values("id").reset_index(drop=True)
    stats["final_rows"] = len(out)
    stats["label_counts"] = {"benign": int((out.label == 0).sum()), "phishing": int(out.label.sum())}
    stats["n_domains"] = int(out.reg_domain.nunique())
    # Report the dataset's structural quirks so artefacts can be discussed in the report.
    has_path = out.url.str.replace(r"^[a-z]+://", "", regex=True).str.contains("/.", regex=True)
    stats["share_with_path_by_label"] = {
        "benign": float(has_path[out.label == 0].mean()),
        "phishing": float(has_path[out.label == 1].mean()),
    }
    stats["share_https_by_label"] = {
        "benign": float(out.url[out.label == 0].str.startswith("https").mean()),
        "phishing": float(out.url[out.label == 1].str.startswith("https").mean()),
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
