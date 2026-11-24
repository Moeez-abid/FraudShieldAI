"""Reproducible train/val/test splits.

* messages: stratified by label.
* urls: grouped by registered domain, so no domain appears in more than one split.

Writes data/processed/<task>_{train,val,test}.parquet and a split manifest with
row IDs and content hashes (data/processed/<task>_split_manifest.json).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

from fraudshield.utils import frame_hash, load_config, resolve, write_json


def stratified_split(df: pd.DataFrame, label_col: str, fracs: dict, seed: int):
    test_val = fracs["val"] + fracs["test"]
    train, rest = train_test_split(df, test_size=test_val, stratify=df[label_col], random_state=seed)
    val, test = train_test_split(
        rest, test_size=fracs["test"] / test_val, stratify=rest[label_col], random_state=seed
    )
    return train, val, test


def grouped_split(df: pd.DataFrame, group_col: str, fracs: dict, seed: int,
                  strata: pd.Series | None = None):
    """Group-disjoint split. With `strata`, uses StratifiedGroupKFold so each split
    keeps roughly the same mix of strata (e.g. label x source); otherwise GroupShuffleSplit.

    StratifiedGroupKFold works in equal folds, so the fractions are approximated with
    k = round(1 / test_frac) folds: fold 0 -> test, fold 1 -> val, rest -> train.
    """
    if strata is not None:
        k = max(3, round(1 / fracs["test"]))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        fold = np.empty(len(df), dtype=int)
        for i, (_, idx) in enumerate(sgkf.split(df, strata, groups=df[group_col])):
            fold[idx] = i
        return df[fold >= 2], df[fold == 1], df[fold == 0]

    test_val = fracs["val"] + fracs["test"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_val, random_state=seed)
    tr_idx, rest_idx = next(gss.split(df, groups=df[group_col]))
    train, rest = df.iloc[tr_idx], df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=fracs["test"] / test_val, random_state=seed)
    va_idx, te_idx = next(gss2.split(rest, groups=rest[group_col]))
    return train, rest.iloc[va_idx], rest.iloc[te_idx]


def save(task: str, splits: dict, cfg: dict, label_col: str, hash_cols: list[str], extra: dict):
    out_dir = resolve(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"task": task, "seed": cfg["seed"], **extra, "splits": {}}
    for name, part in splits.items():
        part = part.sort_values("id").reset_index(drop=True)
        part.to_parquet(out_dir / f"{task}_{name}.parquet", index=False)
        manifest["splits"][name] = {
            "rows": len(part),
            "label_counts": {k: int(v) for k, v in part[label_col].value_counts().items()},
            "hash": frame_hash(part, hash_cols),
            "ids": part["id"].tolist(),
        }
        extra_info = ""
        if "source" in part:
            extra_info = " " + str(part.groupby("source").size().to_dict())
        print(f"  {task}_{name}: {len(part)} rows {manifest['splits'][name]['label_counts']}{extra_info}")
    write_json(manifest, out_dir / f"{task}_split_manifest.json")


def split_messages() -> None:
    cfg = load_config("message")
    df = pd.read_parquet(resolve(cfg["paths"]["interim"]))
    train, val, test = stratified_split(df, "label", cfg["split"], cfg["seed"])
    save("messages", {"train": train, "val": val, "test": test}, cfg, "label",
         ["id", "text", "label"], {"strategy": "stratified"})


def split_urls() -> None:
    cfg = load_config("url")
    df = pd.read_parquet(resolve(cfg["paths"]["interim"]))
    strata = df.source + "/" + df.label.astype(str)
    train, val, test = grouped_split(df, "reg_domain", cfg["split"], cfg["seed"], strata)
    overlap = set(train.reg_domain) & (set(val.reg_domain) | set(test.reg_domain))
    assert not overlap, f"domain leakage across splits: {list(overlap)[:5]}"
    save("urls", {"train": train, "val": val, "test": test}, cfg, "label",
         ["id", "url", "label"], {"strategy": "stratified_group_kfold(registered_domain; strata=source/label)",
                                  "n_domains": int(df.reg_domain.nunique())})

    # Leakage reference: a plain random split, used only to show inflation in the report.
    rtrain, rval, rtest = stratified_split(df, "label", cfg["split"], cfg["seed"])
    shared = len(set(rtrain.reg_domain) & set(rval.reg_domain)) / max(1, rval.reg_domain.nunique())
    print(f"  (random split would share {shared:.1%} of val domains with train)")
    out_dir = resolve(cfg["paths"]["processed_dir"])
    np.save(out_dir / "urls_random_val_ids.npy", rval["id"].to_numpy())
    np.save(out_dir / "urls_random_train_ids.npy", rtrain["id"].to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=["messages", "urls"])
    args = ap.parse_args()
    {"messages": split_messages, "urls": split_urls}[args.task]()


if __name__ == "__main__":
    main()
