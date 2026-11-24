"""Random Forest URL classifier on lexical/structural features.

Steps:
  1. Compute (and cache) URL features for each split.
  2. Tune max_depth / min_samples_leaf on a training subsample, scored on validation F1.
  3. Refit on the full training split; report validation metrics and feature importances.
  4. Leakage check: retrain on a random (non-grouped) split and compare validation scores.
Test metrics are computed only with --test.
"""

from __future__ import annotations

import argparse
import itertools

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraudshield.evaluation.metrics import binary_metrics, plot_confusion
from fraudshield.features.url_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _split,
    extract_frame,
)
from fraudshield.utils import git_commit, load_config, resolve, timestamp, write_json


def make_pipeline(cfg: dict, max_depth, min_samples_leaf: int) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("tld", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30,
                              sparse_output=False), CATEGORICAL_FEATURES),
    ])
    rf = RandomForestClassifier(
        n_estimators=cfg["random_forest"]["n_estimators"], max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, class_weight="balanced", n_jobs=-1,
        random_state=cfg["seed"],
    )
    return Pipeline([("pre", pre), ("rf", rf)])


def view_urls(urls: pd.Series, view: str) -> pd.Series:
    """'full' = the URL as given; 'host' = host name only (no scheme, path or query).

    The host view exists because every benign PhiUSIIL URL is an https homepage, so
    scheme/path features act as a dataset shortcut. Host-only features can't use it.
    """
    if view == "full":
        return urls
    return urls.map(lambda u: _split(u)[1])


def features_for(df: pd.DataFrame, cache_path, view: str = "full") -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if len(cached) == len(df) and (cached["id"].to_numpy() == df["id"].to_numpy()).all():
            return cached.drop(columns="id")
    feats = extract_frame(view_urls(df.url, view))
    feats.insert(0, "id", df["id"].to_numpy())
    feats.to_parquet(cache_path, index=False)
    return feats.drop(columns="id")


def load(cfg: dict, name: str, view: str):
    d = resolve(cfg["paths"]["processed_dir"])
    df = pd.read_parquet(d / f"urls_{name}.parquet")
    return df, features_for(df, d / f"urls_{name}_features_{view}.parquet", view)


def feature_importances(model: Pipeline, top: int = 15) -> dict:
    names = model.named_steps["pre"].get_feature_names_out()
    imp = model.named_steps["rf"].feature_importances_
    order = np.argsort(imp)[::-1][:top]
    return {names[i].split("__", 1)[-1]: float(imp[i]) for i in order}


def leakage_check(cfg: dict, params: dict, view: str) -> dict:
    """Same model on a random split vs the domain-grouped split (validation only)."""
    d = resolve(cfg["paths"]["processed_dir"])
    full = pd.concat([pd.read_parquet(d / f"urls_{s}.parquet") for s in ("train", "val", "test")])
    feats = pd.concat([load(cfg, s, view)[1] for s in ("train", "val", "test")])
    feats.index = full["id"].to_numpy()
    labels = pd.Series(full.label.to_numpy(), index=full["id"].to_numpy())
    tr_ids, va_ids = np.load(d / "urls_random_train_ids.npy", allow_pickle=True), \
        np.load(d / "urls_random_val_ids.npy", allow_pickle=True)
    model = make_pipeline(cfg, **params).fit(feats.loc[tr_ids], labels.loc[tr_ids])
    return binary_metrics(labels.loc[va_ids], model.predict_proba(feats.loc[va_ids])[:, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", action="store_true", help="also evaluate on the held-out test set")
    ap.add_argument("--skip-leakage-check", action="store_true")
    ap.add_argument("--view", choices=["full", "host"], default="full",
                    help="features from the full URL or the host name only")
    args = ap.parse_args()

    cfg = load_config("url")
    rf_cfg = cfg["random_forest"]
    view = args.view
    train, Xtr = load(cfg, "train", view)
    val, Xva = load(cfg, "val", view)
    ytr, yva = train.label.to_numpy(), val.label.to_numpy()

    rng = np.random.default_rng(cfg["seed"])
    sub = rng.choice(len(train), size=min(rf_cfg["tune_sample"], len(train)), replace=False)
    search = {}
    for depth, leaf in itertools.product(rf_cfg["max_depth_grid"], rf_cfg["min_samples_leaf_grid"]):
        m = make_pipeline(cfg, depth, leaf).fit(Xtr.iloc[sub], ytr[sub])
        f1 = binary_metrics(yva, m.predict_proba(Xva)[:, 1])["f1"]
        search[f"depth={depth},leaf={leaf}"] = f1
        print(f"[url_rf] max_depth={depth} min_samples_leaf={leaf}: val F1={f1:.4f}")
    best_key = max(search, key=search.get)
    depth_s, leaf_s = (kv.split("=")[1] for kv in best_key.split(","))
    params = {"max_depth": None if depth_s == "None" else int(depth_s), "min_samples_leaf": int(leaf_s)}

    model = make_pipeline(cfg, **params).fit(Xtr, ytr)
    results = {
        "model": f"url_random_forest_{view}",
        "view": view,
        "best_params": params,
        "search_val_f1": search,
        "val": binary_metrics(yva, model.predict_proba(Xva)[:, 1]),
        "feature_importances": feature_importances(model),
    }
    if args.test:
        test, Xte = load(cfg, "test", view)
        results["test"] = binary_metrics(test.label, model.predict_proba(Xte)[:, 1])
    if not args.skip_leakage_check:
        results["leakage_check_random_split_val"] = leakage_check(cfg, params, view)

    model_dir = resolve(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"url_rf_{view}.joblib", compress=3)

    stamp = timestamp()
    results.update({"timestamp": stamp, "git_commit": git_commit(),
                    "train_rows": len(train), "seed": cfg["seed"]})
    write_json(results, resolve("reports/metrics") / f"url_rf_{view}_{stamp}.json")
    for split in [s for s in ("val", "test") if s in results]:
        cm = results[split]["confusion_matrix"]
        plot_confusion(cm["matrix"], cm["labels"], f"URL RF, {view} view ({split})",
                       resolve("reports/figures") / f"cm_url_rf_{view}_{split}.png")

    v = results["val"]
    print(f"[url_rf:{view}] best {params} | val P={v['precision']:.4f} R={v['recall']:.4f} "
          f"F1={v['f1']:.4f} FPR={v['fpr']:.4f} ROC-AUC={v['roc_auc']:.4f}")
    if "leakage_check_random_split_val" in results:
        lc = results["leakage_check_random_split_val"]
        print(f"[url_rf] random-split val F1={lc['f1']:.4f} (grouped {v['f1']:.4f})")
    print("[url_rf] top features:", list(results["feature_importances"])[:8])


if __name__ == "__main__":
    main()
