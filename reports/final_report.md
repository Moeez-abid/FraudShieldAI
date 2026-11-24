# FraudShield AI: Final Report (draft)

**A Machine Learning System for English Scam Message and Malicious URL Detection**
GenAI Project, Fall 2026 · FAST-NUCES, Department of Software Engineering
Moeez Abid (23i-3033) · Syed Muhammad Ijtaba Rizvi (23i-0112) · Instructor: Shahela Saif

> Draft status: sections marked **[TBD]** need the one-time test evaluation (`make eval`), the labelled challenge set, or the survey results.

---

## 1. Summary
FraudShield AI is a web prototype that gives a second opinion on suspicious English messages and links. A user pastes an SMS, chat message, email excerpt or URL. The app returns an overall **Low / Medium / High** risk rating, a separate verdict for the message and for each link, and the reasons behind each one. Three trained models make the predictions: a **fine-tuned DistilBERT** message classifier, a **TF-IDF + Logistic Regression** baseline, and a **Random Forest** URL classifier on lexical and structural features. Explanations come from word occlusion (messages) and SHAP (links), with rule-based scam signals shown separately. Nothing the user enters is stored, and links are never opened.

The main technical finding was a **dataset shortcut** in the largest public phishing-URL dataset (PhiUSIIL). Every benign URL there is an `https://www.` homepage, so a model trained on raw URLs reaches F1 0.98 but flags **57 %** of real legitimate deep links. Adding a second dataset and training on a "mixed" feature view brings that down to **12 %**.

## 2. Problem and evidence
Phishing campaigns impersonate trusted organisations and push people to hand over credentials or follow malicious links. APWG recorded about 3.8 million phishing attacks in 2025 [1]. Spam filters are tied to individual platforms. FraudShield offers a platform-independent manual check with explanations.

## 3. Users and validation
- **Primary users:** English-reading smartphone and desktop users who want a second opinion.
- **Secondary users:** small-business and university staff handling payment or verification requests.
- **Validation survey** (`docs/survey.md`): **[TBD: n respondents, usefulness and trust medians, willingness to pay, key free-text themes]**.

## 4. System overview

```
pasted text → normalise (NFKC, HTML unescape, ≤2,000 chars)
            → extract URLs (incl. defanged hxxp / [.], bare domains, IPs)
            ├─ message → DistilBERT → band + word occlusion + rule-based signals
            └─ each URL → Random Forest → band + SHAP reasons (+ shortener rule)
            → overall band = highest component band → verdict, headline, advice
```

| Component | Implementation |
|---|---|
| Message model | DistilBERT fine-tuned (3 classes); baseline TF-IDF + LR |
| URL model | Random Forest, 31 URL-text features, no network access |
| Explanations | Log-odds word occlusion; scam-signal regex rules; SHAP TreeExplainer |
| Risk bands | Thresholds tuned on validation (precision target for High, false-alarm cap for Medium) |
| App | Streamlit; HTML-escaped output, no clickable links, usage statistics off |
| Engineering | Python 3.11, scikit-learn, Hugging Face Transformers, pytest (63 tests), Makefile, fixed seeds, split manifests with hashes |

## 5. Data

| Dataset | Use | Size after cleaning |
|---|---|---|
| UCI SMS Spam Collection [2] | ham / spam | merged |
| Mendeley SMS Phishing (Mishra & Soni, 2022) [4] | ham / spam / smishing → scam | **5,875 messages** (4,959 ham / 469 spam / 447 scam) |
| PhiUSIIL (Prasad & Chandra, 2024) [5] | benign / phishing URLs | 235,291 |
| Hannousse & Yahiouche (2021) [6] | benign / phishing full URLs | 11,424 |

**Messages.** The Mendeley set reuses most UCI messages: 5,670 duplicates were removed before splitting. 265 messages labelled "spam" by UCI and "smishing" by Mendeley became `scam`. The split is stratified 70/15/15.

**URLs: shortcut and fix.**

| | has path | https | starts with www. |
|---|---|---|---|
| PhiUSIIL benign | 0 % | 100 % | 100 % |
| PhiUSIIL phishing | 28 % | 49 % | 41 % |
| Hannousse benign | 62 % | 44 % | 67 % |
| Hannousse phishing | 82 % | 33 % | 21 % |

The serving model uses PhiUSIIL **host names only, with a leading `www.` removed** (both classes), plus Hannousse **full URLs**. Path, scheme and query features are therefore learned only where both classes have them.

**Leakage control.** URLs are split by registered domain with StratifiedGroupKFold (strata: source × label), so no domain, and no hosting platform such as `web.app` or `firebaseapp.com`, appears in two splits. A random split would share 9.3 % of validation domains with training.

## 6. Methods
- **Baseline:** word 1–2-gram and char 3–5-gram TF-IDF with Logistic Regression (balanced; C tuned on validation, best 0.3).
- **DistilBERT:** `distilbert-base-uncased` with class-weighted cross-entropy, lr 3e-5, max length 128, effective batch 16, early stopping on validation macro-F1, 4 epochs. Trained on CPU because Apple MPS ran out of memory on the 8 GB laptop. Task-specific fine-tuning, not pre-training.
- **URL Random Forest:** 300 trees, max depth and minimum leaf size tuned on validation, balanced class weights.
- **Risk bands:** High = lowest score with precision ≥ 0.97 (messages) or 0.95 (URLs). Medium = lowest score flagging ≤ 3 % (messages) or ≤ 15 % (URLs) of benign items. URL thresholds are tuned on the Hannousse validation slice, because the app scores full URLs.
- **App rules:** spam is capped at Medium; link shorteners are at least Medium; a message read as `ham` is never High.

## 7. Results

### 7.1 Validation

**Messages (n = 881)**

| | Baseline | DistilBERT |
|---|---|---|
| Macro-F1 | 0.865 | 0.862 |
| Scam recall | 0.82 | **0.90** |
| Spam recall | 0.76 | 0.69 |
| Spam flagged as malicious | 0.83 | **0.89** |
| Scam flagged as malicious | 1.00 | 1.00 |
| Ham false-positive rate | 0.54 % | 0.54 % |

The two are tied on macro-F1 given the small minority classes (about 70 each). DistilBERT is the primary model because of its higher scam recall, which was the proposal's stated priority.

**URLs: the effect of the shortcut fix**

| Feature view | Overall F1 | Hannousse F1 | Hannousse benign flagged |
|---|---|---|---|
| Full URL (shortcut) | 0.983 | 0.773 | 56.7 % |
| Host only | 0.795 | 0.543 | 58.8 % |
| **Mixed (deployed)** | **0.816** | **0.880** | **12.1 %** |

### 7.2 Test set (one-time, frozen models)
**[TBD after `make eval`]**: messages (baseline, DistilBERT) and URLs (all, Hannousse and PhiUSIIL slices), with confusion matrices in `reports/figures/cm_*_test.png`.

### 7.3 Challenge set (independent, hand-labelled)
**[TBD]**: n, Cohen's κ between annotators, and metrics. The URL candidates (45 OpenPhish phishing, 45 real benign deep links) are ready for labelling. Messages still need to be collected.

### 7.4 Speed
About 6 s to load the models once (cached), then **0.35–1.0 s per analysis** on a laptop CPU (target: under 3 s).

## 8. Error analysis
Full details are in `reports/error_analysis.md`.
1. **No scam message is read as normal** (validation, both models).
2. **Spam/scam label noise.** Premium-rate SMS are labelled inconsistently ("free entry" 9 scam / 7 spam; "150p" 64 / 79), which accounts for most message errors.
3. **Transactional false alarms.** Delivery, top-up and bank notices share vocabulary with scams. This is the main user-facing weakness.
4. **Phishing on ordinary-looking domains** is missed by a text-only URL model.
5. **Shortener blind spot.** Bare `tinyurl.com` scored 0.45; fixed in the app with a rule.
6. **Explanation limit.** Leave-one-out finds no single word in messages packed with scam cues. The app says so explicitly.

## 9. Limitations and ethics
- English only. The training data are older UK/India SMS and 2020–2024 URLs, so new scam styles may be missed.
- Scores are uncalibrated. Only bands are shown.
- False reassurance is the biggest risk: every Low result states it doesn't guarantee safety, and the advice points to official channels.
- Privacy: no storage, no logging, no link visits, telemetry off.
- All datasets are CC BY 4.0 and cited.

## 10. Commercial notes
As planned, the MVP is a free manual checker. Possible routes are a capped free tier, a paid individual plan (history, unlimited checks, family sharing) and a usage-based API for small organisations. These depend on demand and pricing validation from the survey: **[TBD: survey willingness-to-pay summary]**. Natural extensions are a WhatsApp bot and a browser extension.

## 11. Proposal vs delivered

| Proposal commitment | Status |
|---|---|
| Fine-tuned DistilBERT + TF-IDF baseline | Done |
| Random Forest URL classifier, no site visits | Done |
| Explanations for text and URLs | Done (occlusion, SHAP, rule signals) |
| Streamlit app with URL extraction | Done |
| Leakage-aware validation, deduplication | Done (domain-grouped splits, cross-source dedup) |
| Spam and scam reported separately | Done |
| Precision, recall, F1, confusion matrices, FPR | Done |
| Independent challenge set | Tooling and URL candidates ready; **labelling [TBD]** |
| 15–20 user survey | Draft ready; **[TBD]** |
| Calibrated probabilities | Not done: bands are shown instead, as the proposal allowed |

## References
[1] APWG, Phishing Activity Trends Report, Q4 2025.
[2] Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI ML Repository.
[3] Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
[4] Mishra, S. & Soni, D. (2022). SMS Phishing Dataset for Machine Learning and Pattern Recognition. Mendeley Data. doi:10.17632/f45bkkt8pr.1
[5] Prasad, A. & Chandra, S. (2024). PhiUSIIL Phishing URL Dataset. UCI ML Repository. doi:10.24432/C5BC8M
[6] Hannousse, A. & Yahiouche, S. (2021). Web page phishing detection. Mendeley Data. doi:10.17632/c2gw7fy2j4.3
[7] Lundberg, S. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS.
