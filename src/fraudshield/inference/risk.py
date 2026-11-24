"""Risk bands (Low / Medium / High) and threshold selection.

Scores are NOT shown as probabilities (they're uncalibrated). Thresholds are chosen
on the validation set, with per-model targets in configs/*.yaml (`risk:`):
  high   : lowest score where precision >= high_precision (few false "High" alarms)
  medium : lowest score where the share of benign inputs flagged <= medium_max_benign_flagged
           (bounds false alarms; a recall target alone flagged 61 % of benign URLs)
Combined verdict = the highest band of any component (message or any URL).

CLI: `python -m fraudshield.inference.risk` recomputes thresholds for the trained
models and writes models/risk_thresholds.json (+ a copy in reports/metrics).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

BANDS = ("Low", "Medium", "High")


def choose_thresholds(y_true, score, high_precision: float = 0.90,
                      medium_max_benign_flagged: float = 0.10) -> dict:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    precision, _, thr = precision_recall_curve(y_true, score)
    precision = precision[:-1]  # align with thresholds

    ok_high = np.where(precision >= high_precision)[0]
    high = float(thr[ok_high[0]]) if len(ok_high) else 1.0
    benign = np.sort(score[y_true == 0])
    # smallest threshold t with mean(benign >= t) <= target
    k = int(np.floor(medium_max_benign_flagged * len(benign)))
    medium = float(np.nextafter(benign[-k - 1], np.inf)) if k < len(benign) else float(benign[0])
    medium = min(medium, high)

    def stats(t):
        flagged = score >= t
        tp = int((flagged & (y_true == 1)).sum())
        return {"threshold": t,
                "precision": tp / max(1, int(flagged.sum())),
                "recall": tp / max(1, int((y_true == 1).sum())),
                "benign_flagged_rate": float(flagged[y_true == 0].mean())}

    return {"medium": medium, "high": high,
            "targets": {"high_precision": high_precision,
                        "medium_max_benign_flagged": medium_max_benign_flagged},
            "at_medium": stats(medium), "at_high": stats(high)}


def band(score: float, thresholds: dict) -> str:
    if score >= thresholds["high"]:
        return "High"
    if score >= thresholds["medium"]:
        return "Medium"
    return "Low"


def max_band(*bands: str | None) -> str:
    present = [b for b in bands if b]
    return max(present, key=BANDS.index) if present else "Low"


def main() -> None:
    import joblib
    import pandas as pd

    from fraudshield.models.url_rf import features_for
    from fraudshield.utils import load_config, resolve, write_json

    out = {}
    mcfg, ucfg = load_config("message"), load_config("url")
    pdir = resolve(mcfg["paths"]["processed_dir"])
    mdir = resolve(mcfg["paths"]["model_dir"])

    val = pd.read_parquet(pdir / "messages_val.parquet")
    y = (val.label != "ham").astype(int)
    if (mdir / "distilbert" / "fraudshield.json").exists():
        from fraudshield.inference.message_model import DistilBertMessageModel
        msg_model, name = DistilBertMessageModel(mdir / "distilbert"), "distilbert"
    else:
        from fraudshield.inference.message_model import BaselineMessageModel
        msg_model, name = BaselineMessageModel(mdir / "baseline_tfidf_lr.joblib"), "baseline"
    malicious = msg_model.predict_proba(val.text.tolist())
    malicious = 1 - malicious[:, msg_model.labels.index("ham")]
    out["message"] = {"model": name, **choose_thresholds(y, malicious, **mcfg["risk"])}

    uval = pd.read_parquet(pdir / "urls_val.parquet")
    view = ucfg.get("serving_view", "mixed")
    feats = features_for(uval, pdir / f"urls_val_features_{view}.parquet", view)
    rf = joblib.load(mdir / f"url_rf_{view}.joblib")
    uscore = rf.predict_proba(feats)[:, 1]
    keep = np.ones(len(uval), dtype=bool)
    if ucfg.get("risk_slice"):
        keep = (uval.source == ucfg["risk_slice"]).to_numpy()
    out["url"] = {"model": f"url_rf_{view}", "slice": ucfg.get("risk_slice", "all"),
                  "n": int(keep.sum()),
                  **choose_thresholds(uval.label[keep], uscore[keep], **ucfg["risk"])}

    write_json(out, mdir / "risk_thresholds.json")
    write_json(out, resolve("reports/metrics") / "risk_thresholds.json")
    for k, v in out.items():
        print(f"[risk] {k} ({v['model']}): medium>={v['medium']:.3f} high>={v['high']:.3f} | "
              f"at medium P={v['at_medium']['precision']:.3f} R={v['at_medium']['recall']:.3f} "
              f"benign flagged={v['at_medium']['benign_flagged_rate']:.3f} | "
              f"at high P={v['at_high']['precision']:.3f} R={v['at_high']['recall']:.3f}")


if __name__ == "__main__":
    main()
