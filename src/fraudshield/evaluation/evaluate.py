"""Evaluate the SAVED (frozen) models without retraining.

    python -m fraudshield.evaluation.evaluate --split val     # any time
    python -m fraudshield.evaluation.evaluate --split test    # once, after models are frozen

Scores every available model (baseline, DistilBERT, URL RF serving view) on the
chosen split, plus the hand-labelled challenge set when data/challenge/*.csv have
`label_final` filled in. Writes reports/metrics/eval_<split>_<timestamp>.json and
confusion-matrix figures, and per-example error dumps for error analysis.
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from fraudshield.evaluation.metrics import binary_metrics, message_metrics, plot_confusion
from fraudshield.features.url_features import extract_frame
from fraudshield.inference.extract_urls import refang
from fraudshield.inference.risk import band
from fraudshield.utils import git_commit, load_config, resolve, timestamp, write_json


def message_models(mdir):
    from fraudshield.inference.message_model import BaselineMessageModel

    models = {"baseline": BaselineMessageModel(mdir / "baseline_tfidf_lr.joblib")}
    if (mdir / "distilbert" / "fraudshield.json").exists():
        try:
            from fraudshield.inference.message_model import DistilBertMessageModel
            models["distilbert"] = DistilBertMessageModel(mdir / "distilbert")
        except ImportError:
            print("[evaluate] torch/transformers not installed; skipping DistilBERT")
    return models


def eval_messages(models, df: pd.DataFrame, labels, tag: str, out_dir) -> dict:
    res = {}
    for name, m in models.items():
        probs = m.predict_proba(df.text.tolist())
        pred = [m.labels[i] for i in probs.argmax(1)]
        res[name] = message_metrics(df.label, pred, labels)
        cm = res[name]["confusion_matrix"]
        plot_confusion(cm["matrix"], cm["labels"], f"{name} ({tag})",
                       resolve("reports/figures") / f"cm_{name}_{tag}.png")
        errors = df.assign(pred=pred, p_ham=probs[:, m.labels.index("ham")])
        errors[errors.label != errors.pred].to_csv(out_dir / f"errors_{name}_{tag}.csv", index=False)
    return res


def eval_urls(rf, df: pd.DataFrame, url_col: str, tag: str, thresholds, out_dir,
              add_scheme: bool = False) -> dict:
    """add_scheme=True mimics the app (extract_urls adds http:// to bare links); use it
    for the challenge set. Dataset splits are scored on the exact training strings."""
    urls = df[url_col]
    if add_scheme:
        urls = urls.map(lambda u: u if "://" in u else "http://" + u)
    score = rf.predict_proba(extract_frame(urls))[:, 1]
    res = {"all": binary_metrics(df.label, score)}
    if "source" in df:
        for src in sorted(df.source.unique()):
            m = (df.source == src).to_numpy()
            res[src] = binary_metrics(df.label[m], score[m])
    if thresholds:
        bands = [band(s, thresholds) for s in score]
        res["band_counts"] = (pd.crosstab(df.label.map({0: "benign", 1: "phishing"}), pd.Series(bands, name="band"))
                              .to_dict())
    cm = res["all"]["confusion_matrix"]
    plot_confusion(cm["matrix"], cm["labels"], f"URL RF ({tag})",
                   resolve("reports/figures") / f"cm_url_rf_{tag}.png")
    wrong = df.assign(score=score)
    wrong = wrong[(wrong.score >= 0.5).astype(int) != wrong.label]
    wrong.sort_values("score").to_csv(out_dir / f"errors_url_rf_{tag}.csv", index=False)
    return res


def load_challenge():
    d = resolve("data/challenge")
    out = {}
    msgs = pd.read_csv(d / "messages.csv", dtype=str).dropna(subset=["label_final"])
    if len(msgs):
        out["messages"] = msgs.rename(columns={"label_final": "label"})[["id", "text", "label"]]
    urls = pd.read_csv(d / "urls.csv", dtype=str).dropna(subset=["label_final"])
    if len(urls):
        urls = urls.assign(url=urls.url_defanged.map(refang),
                           label=urls.label_final.str.strip().str.lower().eq("phishing").astype(int))
        out["urls"] = urls[["id", "url", "label"]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--no-challenge", action="store_true")
    args = ap.parse_args()

    mcfg, ucfg = load_config("message"), load_config("url")
    pdir, mdir = resolve(mcfg["paths"]["processed_dir"]), resolve(mcfg["paths"]["model_dir"])
    stamp = timestamp()
    out_dir = resolve("reports/errors") / f"{args.split}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    thr_path = mdir / "risk_thresholds.json"
    thresholds = __import__("json").loads(thr_path.read_text()) if thr_path.exists() else {}

    results = {"split": args.split, "timestamp": stamp, "git_commit": git_commit()}
    msg_models = message_models(mdir)
    msgs = pd.read_parquet(pdir / f"messages_{args.split}.parquet")
    results["messages"] = eval_messages(msg_models, msgs, mcfg["labels"], args.split, out_dir)

    view = ucfg.get("serving_view", "mixed")
    rf = joblib.load(mdir / f"url_rf_{view}.joblib")
    urls = pd.read_parquet(pdir / f"urls_{args.split}.parquet")
    # Score the same strings the model was trained on (feature_url): full URLs for
    # Hannousse, host-only for PhiUSIIL, so numbers match training-time validation.
    results["urls"] = eval_urls(rf, urls, "feature_url", args.split, thresholds.get("url"), out_dir)

    if not args.no_challenge:
        ch = load_challenge()
        if "messages" in ch:
            results["challenge_messages"] = eval_messages(msg_models, ch["messages"], mcfg["labels"],
                                                          "challenge", out_dir)
        if "urls" in ch:
            results["challenge_urls"] = eval_urls(rf, ch["urls"], "url", "challenge",
                                                  thresholds.get("url"), out_dir, add_scheme=True)
        if not ch:
            print("[evaluate] challenge set has no labelled rows yet; skipped")

    write_json(results, resolve("reports/metrics") / f"eval_{args.split}_{stamp}.json")
    for name, m in results["messages"].items():
        print(f"[evaluate:{args.split}] {name}: macro-F1={m['macro_f1']:.4f} "
              f"ham FPR={m['fpr_benign_flagged']:.4f} "
              f"scam R={m['per_class']['scam']['recall']:.3f}")
    for k, m in results["urls"].items():
        if k != "band_counts":
            print(f"[evaluate:{args.split}] url_rf/{k}: F1={m['f1']:.4f} FPR={m['fpr']:.4f} n={m['n']}")
    print(f"[evaluate] errors -> {out_dir}")


if __name__ == "__main__":
    main()
