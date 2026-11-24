# FraudShield AI

A machine-learning prototype that checks suspicious English messages and URLs and explains its verdict.
FAST-NUCES GenAI project, Fall 2026: Moeez Abid (23i-3033) and Syed Muhammad Ijtaba Rizvi (23i-0112).

- **Message model:** fine-tuned DistilBERT (primary) with a TF-IDF + Logistic Regression fallback. Classes are `ham` / `spam` / `scam`.
- **URL model:** Random Forest on lexical and structural URL features, trained on a "mixed" view that removes PhiUSIIL's shortcuts. URLs are never opened or fetched.
- **Explanations:** word occlusion (log-odds) and rule-based scam signals for messages; SHAP for URLs. Low / Medium / High risk bands are tuned on validation.
- **App:** Streamlit. Paste a message or link and get an overall Low / Medium / High verdict, highlighted words, scam signals and per-link reasons. Nothing is stored, and links are never opened or made clickable.

Progress: [Week 1](reports/week1_summary.md) · [Week 2](reports/week2_summary.md) · [Week 3](reports/week3_summary.md)

## Run the app

Needs the trained models in `models/` (from `make train`, or copied from the shared Drive folder):

```bash
make app
```

Then open http://localhost:8501. Without torch/transformers installed, the app falls back to the TF-IDF baseline. In that case, run `make thresholds` first so the risk bands match the model in use.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full plan.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (or any Python 3.11 virtualenv).

```bash
make setup
```

The DistilBERT dependencies are kept separate because they are large (about 450 MB). Install them on Colab or locally when needed:

```bash
make setup-dl
```

For DistilBERT training, Colab is easiest: open [notebooks/train_distilbert_colab.ipynb](notebooks/train_distilbert_colab.ipynb) with a T4 runtime and **Run all**. It runs the hyperparameter sweep and downloads `distilbert_sweep_results.zip`. On an 8 GB Mac, Apple MPS runs out of memory; train on the CPU instead (about 45 minutes):

```bash
make train-distilbert DISTILBERT_ARGS="--cpu --batch-size 8 --grad-accum 2"
```

## Reproduce

```bash
make data     # download, clean, deduplicate, split (writes data/processed + split manifests)
make train    # baseline, URL RF (mixed view), DistilBERT, risk thresholds
make train-url-compare   # full-URL and host-only URL models, for the report
make test     # unit tests (no trained models needed)
make eval-val # score the saved models on validation (+ challenge set when labelled)
make eval     # score the saved models on the held-out TEST split: only once models are frozen
```

Seeds are fixed in `configs/*.yaml`. Each split manifest (`data/processed/*_split_manifest.json`) stores row IDs and a content hash, so a split can be checked for exact reproduction.

## After the Colab sweep

1. Keep the current model in case the sweep isn't better:

   ```bash
   mv models/distilbert models/distilbert_cpu_backup
   ```

2. Unzip the results into the repo root. This creates `models/distilbert/` and adds the run's metrics JSON and confusion matrix:

   ```bash
   unzip -o ~/Downloads/distilbert_sweep_results.zip
   ```

3. Compare the new run (printed at the end of the notebook, and in `reports/metrics/distilbert_*.json`) with the current one: **val macro-F1 0.862, scam recall 0.896, ham FPR 0.54 %**. Keep whichever is better on scam recall without a worse ham FPR. To go back to the old model, delete `models/distilbert` and rename the backup.
4. Recompute the risk bands for the chosen model and check it on validation:

   ```bash
   make thresholds eval-val
   ```

5. Freeze. No more model or threshold changes after this. Then run the **one-time** test evaluation:

   ```bash
   make eval
   ```

## Sharing with a teammate (Google Drive)

Data and trained models are **not in git**: they're too big for a slow HTTPS push, and the derived tables change on every rebuild. Share these through Drive, keeping the same paths:

| Path | Size | Needed? |
|---|---|---|
| `models/distilbert/` | 256 MB | To run the app without retraining DistilBERT (about 45 min on CPU) |
| `models/url_rf_mixed.joblib` | 45 MB | To run the app without retraining the URL model (about 1 min) |
| `models/baseline_tfidf_lr.joblib` | 2 MB | Fallback message model |
| `models/risk_thresholds.json` | 4 KB | Risk bands (or rerun `make thresholds`) |
| `data/raw/phiusiil/phiusiil.zip` | 15 MB | Recommended: the UCI server is very slow and drops connections |
| `data/raw/uci_sms/uci_sms.zip`, `data/raw/mendeley_smishing/mendeley_smishing.zip`, `data/raw/hannousse/dataset_B_05_2020.csv` | 4 MB | Optional (fast to download) |

Not needed: `.venv/` (run `make setup` and `make setup-dl`), `data/interim/`, `data/processed/*.parquet` and the comparison models `url_rf_full` and `url_rf_host` (`make train-url-compare`).

To build the bundle:

```bash
zip -r fraudshield_share.zip models/distilbert models/url_rf_mixed.joblib models/baseline_tfidf_lr.joblib models/risk_thresholds.json data/raw/phiusiil/phiusiil.zip data/raw/uci_sms/uci_sms.zip data/raw/mendeley_smishing/mendeley_smishing.zip data/raw/hannousse/dataset_B_05_2020.csv -x "models/distilbert/training_args.bin"
```

On the other machine: clone, unzip the bundle into the repo root, then run `make setup setup-dl data`. `make data` uses the archives already in `data/raw/` instead of downloading them, and rebuilds the cleaned tables and splits. The split hashes in `data/processed/*_split_manifest.json` should match the committed ones. The saved models need the pinned `scikit-learn==1.5.2` from `requirements.txt` to load.

## Data

| Dataset | Use | Licence |
|---|---|---|
| UCI SMS Spam Collection (Almeida & Hidalgo, 2011) | ham / spam messages | CC BY 4.0 |
| Mendeley SMS Phishing Dataset (Mishra & Soni, 2022), doi:10.17632/f45bkkt8pr.1 | ham / spam / smishing → scam | CC BY 4.0 |
| PhiUSIIL Phishing URL Dataset (Prasad & Chandra, 2024), doi:10.24432/C5BC8M | benign / phishing URLs (URL string only) | CC BY 4.0 |
| Web page phishing detection (Hannousse & Yahiouche, 2021), doi:10.17632/c2gw7fy2j4.3 | benign / phishing full URLs, including deep links | CC BY 4.0 |

Notes:
- The Mendeley set reuses most of the UCI messages. Merging and deduplicating them leaves 5,875 unique messages: 4,959 ham, 469 spam and 447 scam. Where UCI said "spam" and Mendeley said "smishing" for the same message (265 cases), the message is labelled `scam`.
- URLs are split **by registered domain** (StratifiedGroupKFold on source × label), so no domain or hosting platform appears in more than one split. A random-split score is also reported to show how much domain leakage would inflate results.
- **PhiUSIIL artefacts:** every benign URL is an `https://www.` homepage with no path. The serving model uses PhiUSIIL **host names only, with a leading `www.` removed**, plus **full URLs from Hannousse**, where both classes have deep links. See [reports/week2_summary.md](reports/week2_summary.md).
- Raw data is git-ignored and re-downloaded by `make data` (or copied from Drive, see above). `data/raw/MANIFEST.json` records the source URL and SHA-256 of each download.

## Layout

```
configs/            hyperparameters, paths, label maps
src/fraudshield/
  data/             download, clean, split
  features/         URL feature extraction
  models/           baseline, DistilBERT, URL RF
  explain/          word occlusion, scam-signal rules, URL SHAP
  inference/        URL extraction, model wrappers, risk bands, analyse() pipeline
  evaluation/       metrics + evaluate.py (frozen-model evaluation, error dumps)
notebooks/          Colab notebook for DistilBERT
docs/               survey draft
app/                Streamlit app
reports/            metrics JSON + figures
tests/              pytest suite
```
