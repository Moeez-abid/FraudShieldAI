"""TF-IDF (word + char n-grams) + Logistic Regression message baseline.

Tunes C on the validation set (macro-F1), refits on train, and reports validation
metrics. Test metrics are only computed with --test, so the test set is not used
for model selection.
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from fraudshield.evaluation.metrics import message_metrics, plot_confusion
from fraudshield.utils import git_commit, load_config, resolve, timestamp, write_json


def make_pipeline(cfg: dict, C: float) -> Pipeline:
    b = cfg["baseline"]
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=tuple(b["word_ngram"]), min_df=b["min_df"],
                                 sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(b["char_ngram"]),
                                 min_df=b["min_df"], sublinear_tf=True, lowercase=True)),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=b["max_iter"],
                             random_state=cfg["seed"])
    return Pipeline([("tfidf", features), ("clf", clf)])


def load_split(cfg: dict, name: str) -> pd.DataFrame:
    return pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / f"messages_{name}.parquet")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", action="store_true", help="also evaluate on the held-out test set")
    args = ap.parse_args()

    cfg = load_config("message")
    labels = cfg["labels"]
    train, val = load_split(cfg, "train"), load_split(cfg, "val")

    search = {}
    for C in cfg["baseline"]["C_grid"]:
        pipe = make_pipeline(cfg, C).fit(train.text, train.label)
        m = message_metrics(val.label, pipe.predict(val.text), labels)
        search[str(C)] = m["macro_f1"]
        print(f"[baseline] C={C}: val macro-F1={m['macro_f1']:.4f}")
    best_C = float(max(search, key=search.get))

    model = make_pipeline(cfg, best_C).fit(train.text, train.label)
    results = {
        "model": "tfidf_logreg",
        "best_C": best_C,
        "search_val_macro_f1": search,
        "val": message_metrics(val.label, model.predict(val.text), labels),
    }
    if args.test:
        test = load_split(cfg, "test")
        results["test"] = message_metrics(test.label, model.predict(test.text), labels)

    model_dir = resolve(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "baseline_tfidf_lr.joblib")

    stamp = timestamp()
    results.update({"timestamp": stamp, "git_commit": git_commit(),
                    "train_rows": len(train), "seed": cfg["seed"]})
    write_json(results, resolve("reports/metrics") / f"baseline_tfidf_lr_{stamp}.json")
    for split in [s for s in ("val", "test") if s in results]:
        cm = results[split]["confusion_matrix"]
        plot_confusion(cm["matrix"], cm["labels"], f"TF-IDF + LR ({split})",
                       resolve("reports/figures") / f"cm_baseline_{split}.png")

    v = results["val"]
    print(f"[baseline] best C={best_C} | val macro-F1={v['macro_f1']:.4f} "
          f"| ham FPR={v['fpr_benign_flagged']:.4f}")
    for lab, pc in v["per_class"].items():
        print(f"  {lab:5s} P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f} n={pc['support']}")
    for name, sl in v["slices"].items():
        print(f"  {name}: flagged recall={sl['recall_flagged']:.3f} exact-class recall={sl['recall_exact_class']:.3f}")


if __name__ == "__main__":
    main()
