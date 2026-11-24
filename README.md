# FraudShield AI

A machine-learning prototype that checks suspicious English messages and URLs and explains its verdict.
FAST-NUCES GenAI project, Fall 2026: Moeez Abid (23i-3033) and Syed Muhammad Ijtaba Rizvi (23i-0112).

- **Message model:** TF-IDF + Logistic Regression baseline, then fine-tuned DistilBERT (Week 2). Classes are `ham` / `spam` / `scam`.
- **URL model:** Random Forest on lexical and structural URL features. URLs are never opened or fetched.
- **App:** Streamlit (Week 3).

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the full plan.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (or any Python 3.11 virtualenv).

```bash
make setup
```

The DistilBERT dependencies are kept separate because they are large. Install them on Colab or locally when needed:

```bash
uv pip install --python .venv/bin/python -r requirements-dl.txt
```

## Reproduce

```bash
make data     # download, clean, deduplicate, split (writes data/processed + split manifests)
make train    # baseline + URL Random Forest; metrics to reports/metrics, figures to reports/figures
make test     # unit tests
make eval     # final held-out test evaluation: run only after models are frozen
```

Seeds are fixed in `configs/*.yaml`. Each split manifest (`data/processed/*_split_manifest.json`) stores row IDs and a content hash, so a split can be checked for exact reproduction.

## Data

| Dataset | Use | Licence |
|---|---|---|
| UCI SMS Spam Collection (Almeida & Hidalgo, 2011) | ham / spam messages | CC BY 4.0 |
| Mendeley SMS Phishing Dataset (Mishra & Soni, 2022), doi:10.17632/f45bkkt8pr.1 | ham / spam / smishing → scam | CC BY 4.0 |
| PhiUSIIL Phishing URL Dataset (Prasad & Chandra, 2024), doi:10.24432/C5BC8M | benign / phishing URLs (URL string only) | CC BY 4.0 |

Notes:
- The Mendeley set reuses most of the UCI messages. Merging and deduplicating them leaves 5,875 unique messages: 4,959 ham, 469 spam and 447 scam. Where UCI said "spam" and Mendeley said "smishing" for the same message (265 cases), the message is labelled `scam`.
- URLs are split **by registered domain**, so no domain appears in more than one split. A random-split score is also reported to show how much domain leakage would inflate results.
- **Known PhiUSIIL artefact:** every benign URL is an https homepage with no path. `url_rf --view host` trains on the host name only, which removes that shortcut. See [reports/week1_summary.md](reports/week1_summary.md).
- Raw data is git-ignored and re-downloaded by `make data`. `data/raw/MANIFEST.json` records the SHA-256 of each download.

## Layout

```
configs/            hyperparameters, paths, label maps
src/fraudshield/
  data/             download, clean, split
  features/         URL feature extraction
  models/           baseline, URL RF (DistilBERT in Week 2)
  evaluation/       shared metric functions
  explain/          (Week 2)
  inference/        (Week 3)
app/                Streamlit app (Week 3)
reports/            metrics JSON + figures
tests/              pytest suite
```
