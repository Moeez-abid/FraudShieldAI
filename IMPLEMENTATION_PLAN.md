# FraudShield AI — Implementation Plan

Source: GenAI Project Proposal (Moeez Abid 23i-3033, Syed Muhammad Ijtaba Rizvi 23i-0112), Fall 2026.
Duration: 4 weeks. Delivery: Streamlit web app + reproducible training/evaluation code + report.

---

## 1. Definition of done

The project is finished when all of these are true:

1. A user can paste a message or URL into the Streamlit app and get a separate risk label and reasons for the text and for each extracted URL, in under 3 s on CPU.
2. Three trained models exist and are versioned: TF-IDF + Logistic Regression (baseline), fine-tuned DistilBERT (messages), and a Random Forest (URLs).
3. One command reproduces every reported metric from raw data, with fixed seeds and fixed splits.
4. The report covers precision, recall, F1, confusion matrix and false-positive rate on the test set. Spam and scam are reported separately, plus results on an independently labelled challenge set.
5. Failure cases are documented (at least 10 per model) and there are 15–20 survey responses.
6. Raw input is never written to disk and URLs are never fetched or opened.

---

## 2. Decisions to confirm (defaults chosen)

| Question | Default | Why |
|---|---|---|
| Message label scheme | **3 classes: `ham` / `spam` / `scam`** | Satisfies "report spam and scam separately" directly. Spam ≠ fraud. |
| Scam-message supplement | **Mendeley "SMS Phishing Dataset" (Mishra & Soni, 2022)**: ham / spam / smishing | Public, labelled smishing in English. Check licence and overlap with UCI before use. |
| URL dataset | **PhiUSIIL Phishing URL Dataset (UCI, 2024, ~235k URLs, CC BY 4.0)** | Recent, documented provenance, balanced classes. Only the raw URL string and label are used, not its precomputed page-content features. |
| Compute | **Google Colab T4 GPU** for DistilBERT; everything else on a laptop CPU | Fine-tuning ~10k short messages takes minutes on a T4. |
| Combined verdict | **Overall = highest component risk** (documented rule, no averaging) | Conservative, simple and explainable. |
| Score display | Risk bands (Low / Medium / High), not percentages | The proposal says scores aren't shown as probabilities unless calibration is tested. |

---

## 3. Repository structure

```
FraudShieldAI/
├── README.md
├── IMPLEMENTATION_PLAN.md
├── requirements.txt
├── Makefile                    # make data / train / eval / app
├── configs/
│   ├── message.yaml            # paths, seeds, hyperparameters
│   └── url.yaml
├── data/
│   ├── raw/                    # downloaded, never edited (git-ignored)
│   ├── interim/                # cleaned, deduplicated
│   ├── processed/              # train/val/test parquet + split manifest
│   └── challenge/              # hand-labelled challenge set (committed)
├── src/fraudshield/
│   ├── data/
│   │   ├── download.py
│   │   ├── clean_messages.py   # normalise, dedup, merge sources
│   │   ├── clean_urls.py
│   │   └── split.py            # stratified / domain-grouped splits
│   ├── features/
│   │   └── url_features.py     # lexical + structural features
│   ├── models/
│   │   ├── baseline.py         # TF-IDF + LR
│   │   ├── distilbert.py       # HF Trainer fine-tuning
│   │   └── url_rf.py
│   ├── explain/
│   │   ├── text_occlusion.py
│   │   ├── text_signals.py     # heuristic signal tags (labelled as such)
│   │   └── url_shap.py
│   ├── inference/
│   │   ├── extract_urls.py
│   │   ├── risk.py             # thresholds → bands, combined rule
│   │   └── pipeline.py         # single analyse(text) entry point
│   └── evaluation/
│       ├── metrics.py
│       └── report.py           # tables, confusion matrices, error dumps
├── notebooks/                  # EDA only; no logic lives here
├── models/                     # saved artefacts (git-ignored or Git LFS)
├── reports/
│   ├── figures/
│   └── metrics/                # JSON per run
├── app/
│   └── streamlit_app.py
└── tests/
    ├── test_extract_urls.py
    ├── test_url_features.py
    ├── test_risk.py
    └── test_pipeline.py
```

**Stack:** Python 3.11, `transformers`, `datasets`, `torch`, `scikit-learn`, `pandas`, `pyarrow`, `tldextract`, `shap`, `streamlit`, `pytest`, `pyyaml`. Pin exact versions in `requirements.txt` during Week 1.

---

## 4. Data plan

### 4.1 Messages
1. **Sources:** UCI SMS Spam Collection (5,574) and the Mendeley SMS Phishing Dataset. Optionally add a few hundred hand-collected, de-identified scam examples (for example from public scam-awareness pages), recorded with their source in a provenance CSV.
2. **Label mapping:** UCI `ham→ham` and `spam→spam`. Mendeley `ham→ham`, `spam→spam`, `smishing→scam`. Record the mapping in `configs/message.yaml`.
3. **Cleaning:** Unicode NFKC, collapse whitespace, keep case and punctuation (urgency signals live there), and replace URLs/phones with tokens (`<URL>`, `<PHONE>`) **for a second copy only**. Train on both variants and keep the one that does better on validation.
4. **Deduplication:** exact match after normalisation, then near-duplicates (lowercase, strip digits, hash). The two sources overlap, so dedup **before** splitting and log how many rows were removed.
5. **Split:** stratified 70/15/15 by label, seed 42. Save row IDs to `data/processed/split_manifest.json`.

### 4.2 URLs
1. **Source:** PhiUSIIL, `URL` + `label` columns only.
2. **Cleaning:** strip whitespace, drop malformed rows, lowercase the host only, remove exact duplicates.
3. **Leakage control:** use `tldextract` to get the registered domain, then a **GroupShuffleSplit by registered domain** (70/15/15), so no domain appears in more than one split. Also report one random-split score to show how much leakage would have inflated results.
4. **Imbalance:** report class ratios. Use `class_weight="balanced"` rather than resampling.

### 4.3 Challenge set (independent evaluation)
- 60–100 recent English messages (about half scam, a quarter spam, a quarter ham) and 60–100 URLs that are **not** taken from the training sources.
- Both members label each item independently. Report Cohen's κ, resolve disagreements by discussion and store the result in `data/challenge/`.
- Never used for tuning, only for final reporting.

---

## 5. Models

### 5.1 Baseline: TF-IDF + Logistic Regression
- Word 1–2 grams plus character 3–5 grams, `min_df=2`, `sublinear_tf=True`.
- `LogisticRegression(class_weight="balanced", max_iter=2000)`; tune `C ∈ {0.1, 1, 10}` on validation.
- This is also the CPU fallback model if DistilBERT has problems.

### 5.2 DistilBERT message classifier
- `distilbert-base-uncased` with `AutoModelForSequenceClassification` (3 labels).
- `max_length=128`, batch size 16, `lr=2e-5`, weight decay 0.01, warmup 10 %, at most 4 epochs.
- Early stopping (patience 1) on **validation macro-F1**. Use class-weighted cross-entropy (a custom `Trainer.compute_loss`) because scam is the minority class.
- Fixed seeds. Save the best checkpoint, tokenizer and `label2id` to `models/distilbert/`.
- Small sweep at most: lr ∈ {2e-5, 3e-5, 5e-5} × max_length ∈ {64, 128}.

### 5.3 URL Random Forest
Features (`url_features.py`, pure string parsing, **no network calls**):

| Group | Features |
|---|---|
| Length | URL length, host length, path length, query length, number of path segments |
| Characters | counts of `.` `-` `_` `@` `?` `=` `&` `%` `~`, digit ratio, letter ratio, non-ASCII count |
| Host | is an IP address, number of subdomains, TLD (top-N one-hot plus "other"), punycode (`xn--`), port present |
| Content | `https` scheme, suspicious keywords (`login`, `verify`, `secure`, `account`, `update`, `bank`, `free`, `bonus`…), brand name outside the registered domain, `//` redirect in the path |
| Randomness | Shannon entropy of the host and of the path, longest digit run |
| Shorteners | domain is in a known URL-shortener list |

- `RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1)`; tune `max_depth` and `min_samples_leaf` on validation.
- Optional comparison: Logistic Regression on the same features.

### 5.4 Risk bands
- On the **validation** set, pick thresholds for "malicious" probability (spam + scam for messages): **High** where precision ≥ 0.90, **Medium** for the band where recall is maximised, **Low** below that. Save them to `configs/*.yaml`.
- If time allows, check calibration with a reliability diagram and Brier score. Only then could a probability be shown.

---

## 6. Explainability

| Component | Method | Shown to user as |
|---|---|---|
| Message (model-grounded) | **Word occlusion:** remove each word, re-score, and rank by drop in malicious score. Top 5 words are highlighted. | "These words pushed the model toward *scam*." |
| Message (heuristic) | Regex / lexicon tags: urgency ("act now", "within 24 hours"), credential request ("verify your account", "PIN", "OTP"), money/prize, threat, impersonated brand | Shown separately, labelled **"Common scam signals found"** so they aren't mistaken for model reasoning |
| URL | **SHAP TreeExplainer** on the RF, top 3–5 features with direction | "Contains an IP address instead of a domain", "Unusually long subdomain chain", etc. Each feature maps to a plain-English sentence. |

Occlusion costs N+1 forward passes. Batch them and cap at 64 words.

---

## 7. Inference pipeline and app

```
analyse(text):
  text  = normalise(text)
  urls  = extract_urls(text)            # regex + tldextract validation; handles bare domains, hxxp, [.] defang
  msg   = None if text is URL-only else message_model(text) → band + occlusion + signals
  url_results = [url_model(u) → band + SHAP reasons for u in urls[:5]]
  overall = max_band(msg, *url_results)
  return Report(msg, url_results, overall, disclaimer)
```

**Streamlit layout**
1. Text area, sample-message buttons (benign / spam / scam / URL) and an Analyse button.
2. Overall verdict banner (Low / Medium / High) with a one-line reason.
3. Two columns: **Message analysis** (band, highlighted words, signal tags) and **Link analysis** (one card per URL with band and reasons).
4. Fixed advice panel: "Verify through the organisation's official app, website or phone number. A Low rating does not guarantee safety."
5. Expandable "How this works / limitations" section with model metrics.

**Privacy:** no logging of inputs, `@st.cache_resource` for models only, no `requests` calls to analysed URLs, and the privacy statement in the footer.

---

## 8. Evaluation plan

| Item | Messages | URLs |
|---|---|---|
| Metrics | Per-class precision / recall / F1, macro-F1, confusion matrix, FPR on ham | Precision / recall / F1, FPR, ROC-AUC, confusion matrix |
| Comparisons | Baseline vs DistilBERT (same split) | Domain-grouped vs random split; RF vs LR (optional) |
| Separate reporting | Spam vs ham and scam vs ham slices | — |
| Challenge set | Same metrics, small-n caveat | Same metrics |
| Error analysis | 10+ false negatives and 10+ false positives, categorised | Same |
| Performance | p50/p95 latency on CPU | Same |
| Significance | McNemar test baseline vs DistilBERT (optional) | — |

Every run writes `reports/metrics/<model>_<timestamp>.json` with config hash, data split hash and git commit.

**User validation:** a 10-question Google Form for 15–20 people covering scam encounters, usefulness of the reasons (1–5), trust, a would-you-use question and willingness to pay. Include 3 live app trials per participant if possible.

---

## 9. Week-by-week schedule

Owners: **M** = Moeez (message pipeline), **I** = Ijtaba (URL pipeline), **Both** = shared.

### Week 1: Data and baselines
| Day | Task | Owner |
|---|---|---|
| 1 | Repo skeleton, `requirements.txt`, Makefile, configs, git repo | Both |
| 1–2 | Download UCI SMS + Mendeley; EDA notebook (lengths, class balance, overlap) | M |
| 1–2 | Download PhiUSIIL; EDA (TLDs, lengths, domain counts) | I |
| 3 | Message cleaning, label mapping, dedup, stratified split | M |
| 3–4 | URL cleaning, `url_features.py` + unit tests, domain-grouped split | I |
| 4–5 | TF-IDF + LR baseline, metrics module | M |
| 4–5 | Random Forest v1, random vs grouped split comparison | I |
| 5 | Draft survey; start collecting challenge-set items | Both |

**Exit criteria:** fixed splits committed, baseline metrics JSON for both tasks, feature tests passing.

### Week 2: DistilBERT and explanations
| Day | Task | Owner |
|---|---|---|
| 1–2 | DistilBERT training script (Colab), weighted loss, early stopping | M |
| 3 | Small hyperparameter sweep; pick best on validation macro-F1 | M |
| 1–2 | RF tuning; SHAP TreeExplainer; feature→sentence mapping | I |
| 3–4 | Word-occlusion explainer and heuristic signal tagger | M |
| 3–4 | Threshold selection → risk bands for both models (`risk.py` + tests) | I |
| 5 | Evaluation report script (tables + confusion matrices); first test-set numbers | Both |
| 5 | Send out survey | Both |

**Exit criteria:** DistilBERT checkpoint saved and beating or matching the baseline on validation; explanations returned for any input.

### Week 3: Integration and app
| Day | Task | Owner |
|---|---|---|
| 1 | `extract_urls.py` (incl. defanged and bare domains) + tests | I |
| 1–2 | `pipeline.analyse()` combining both models + combined rule | M |
| 2–4 | Streamlit UI: input, verdict banner, two result columns, advice panel | I |
| 3–4 | Word highlighting component, sample inputs, limitations panel | M |
| 4 | End-to-end tests; CPU latency check; ONNX/quantisation only if too slow | Both |
| 5 | Finish labelling challenge set; compute Cohen's κ | Both |

**Exit criteria:** app runs locally from `make app`; end-to-end test passes; latency < 3 s on CPU.

### Week 4: Testing, feedback and demo
| Day | Task | Owner |
|---|---|---|
| 1 | Final test-set and challenge-set evaluation (frozen models) | Both |
| 2 | Error analysis write-up with categorised failure cases | M (messages), I (URLs) |
| 2–3 | Live user trials and survey analysis | Both |
| 3 | Fix UX issues from feedback (no model retraining after test eval) | I |
| 4 | Final report: method, data, results, limitations, ethics, commercial notes | M |
| 4 | README with setup, reproduction steps and model card per model | I |
| 5 | Demo script, slides, rehearsal; optional deploy to Streamlit Community Cloud / HF Spaces | Both |

**Exit criteria:** every item in §1 is done.

---

## 10. Risks (beyond the proposal)

| Risk | Trigger | Response |
|---|---|---|
| Mendeley licence or overlap problem | Licence unclear, or >50 % overlap with UCI | Use UCI only for spam/ham and rely on the challenge set plus hand-collected examples for scam; state this limitation |
| DistilBERT overfits small scam class | Val loss rises after epoch 1, or scam recall < baseline | Freeze lower layers, lower lr, more dropout; fall back to the baseline in the app |
| URL model learns dataset artefacts (e.g. `https` alone separates classes) | One feature dominates SHAP or the score is near-perfect | Drop or ablate that feature and report the ablation |
| App too slow on CPU | p95 > 3 s | Cap occlusion words, cache the tokenizer, dynamic-quantise DistilBERT |
| Colab session limits | Disconnects mid-training | Checkpoint every epoch to Google Drive |
| Survey response shortfall | < 15 by Week 4 Day 2 | Classmates and staff walk-up trials; report actual n |

---

## 11. Final deliverables checklist

- [ ] Source code with tests (`pytest` green)
- [ ] `make data && make train && make eval` reproduces reported numbers
- [ ] Saved models + config + split manifest
- [ ] Streamlit app (local, optionally deployed)
- [ ] Metrics JSON + figures in `reports/`
- [ ] Challenge set with labelling protocol and κ
- [ ] Error-analysis document
- [ ] Survey results summary
- [ ] Final report + model cards + demo slides
