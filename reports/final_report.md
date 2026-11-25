# FraudShield AI: Final Report (draft)

**A Machine Learning System for English Scam Message and Malicious URL Detection**
GenAI Project, Fall 2026 · FAST-NUCES, Department of Software Engineering
Moeez Abid (23i-3033) · Syed Muhammad Ijtaba Rizvi (23i-0112) · Instructor: Shahela Saif

> Draft status: the test-set evaluation is complete (Section 7.2). Sections marked **[TBD]** need the labelled challenge set or the survey results.

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
            → overall band = highest component band
            → optional user context (sender, expected?, account?) → raise-only floor
  (conversation mode: split chat into turns → windows of ≤3 messages / ≤128 tokens → max,
   + cross-message rules, separate thresholds from synthetic validation chats)
            → verdict, headline, advice
```

| Component | Implementation |
|---|---|
| Message model | DistilBERT fine-tuned (3 classes); baseline TF-IDF + LR |
| URL model | Random Forest, 31 URL-text features, no network access |
| Explanations | Log-odds word occlusion; scam-signal regex rules; SHAP TreeExplainer |
| Risk bands | Thresholds tuned on validation (precision target for High, false-alarm cap for Medium) |
| App | Streamlit; HTML-escaped output, no clickable links, usage statistics off |
| Engineering | Python 3.11, scikit-learn, Hugging Face Transformers, pytest (85 tests), Makefile, fixed seeds, split manifests with hashes |

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
- **DistilBERT:** `distilbert-base-uncased` with class-weighted cross-entropy, lr 3e-5, max length 128, effective batch 16, early stopping on validation macro-F1, 4 epochs. A 6-run sweep was run afterwards and the original model kept (Section 7.1). Trained on CPU because Apple MPS ran out of memory on the 8 GB laptop. Task-specific fine-tuning, not pre-training.
- **URL Random Forest:** 300 trees, max depth and minimum leaf size tuned on validation, balanced class weights.
- **Risk bands:** High = lowest score with precision ≥ 0.97 (messages) or 0.95 (URLs). Medium = lowest score flagging ≤ 3 % (messages) or ≤ 15 % (URLs) of benign items. URL thresholds are tuned on the Hannousse validation slice, because the app scores full URLs.
- **App rules:** spam is capped at Medium; link shorteners are at least Medium; a message read as `ham` is never High.
- **Optional context rules** (`inference/context_rules.py`): the user can say who sent the message (saved contact / unknown mobile number / sender name or short code), whether they expected it, and whether they have an account with the organisation named. Rules then set a minimum band with a plain reason, for example an organisation writing from an ordinary mobile number (High), "family on a new number" from an unknown number (Medium; High if it asks for money or a code), an unexpected parcel or payment message that asks for action (High), or any request to send a one-time code (High; negated warnings such as "do not share your OTP" are excluded). Context **only raises** the rating, because contacts can be hacked and sender names spoofed, and it **never reaches the models**. With no context the output is identical to the evaluated system, so the results in Section 7 are unaffected. These rules were added after the test evaluation and are **not evaluated** on any labelled set, since none of the datasets record context.
- **Conversation mode** (`inference/conversation.py`, `conversation_rules.py`): a pasted chat (WhatsApp export, `Name: message` lines, or one message per line) is split into turns, and the user's own messages are left out. Joining a chat into one text fails: DistilBERT reads only 128 tokens, so the end of a long chat (where the scam usually is) is cut off, and friendly small talk dilutes the scam (in one 12-message test chat the joined score was 0.003 while the last two messages together scored 0.995). Instead the other person's messages are scored in **overlapping windows** of up to 3 consecutive messages that fit the 126-token budget, and any single message longer than the budget is cut into overlapping chunks (stride half the budget). The conversation score is the highest window; the same overlap chunking also applies to a single pasted message over 128 tokens (raise-only; no test message is that long, so Section 7 is unaffected). Because the maximum over many windows flags benign chats more often, **separate thresholds** were tuned on 1,100 synthetic chats built only from **validation** messages (Section 7.5). Hand-written **cross-message rules** catch openers and requests that are far apart: 'new number / lost phone' then a money or code request; 'wrong number' then investment talk; an investment pitch then a money request (High if it promises guaranteed or very high profits); 'code sent by mistake' then 'send it to me'; an easy-job offer then a fee; and repeated money or code requests. Like all app rules they only raise the rating.

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

**Hyperparameter sweep (Colab T4; validation only).** Learning rate {2e-5, 3e-5, 5e-5} × max length {64, 128}, effective batch 16, 4 epochs. Validation macro-F1 per run:

| lr \ max length | 64 | 128 |
|---|---|---|
| 2e-5 | 0.876 | 0.870 |
| 3e-5 | 0.860 | 0.850 |
| 5e-5 | 0.867 | **0.882** |

| Validation | Deployed (CPU run, lr 3e-5 / 128) | Sweep best (lr 5e-5 / 128) |
|---|---|---|
| Macro-F1 | 0.862 | 0.882 |
| Scam recall | **0.896** | 0.866 |
| Scam precision | 0.80 | 0.87 |
| Spam recall | 0.69 | 0.80 |
| Ham false-positive rate | **0.54 %** | 1.08 % |
| Scam flagged as malicious | 100 % | 100 % |

**Decision: keep the deployed model.** The selection rule was fixed before the sweep: prefer higher scam recall without a higher ham false-positive rate. The sweep winner is worse on both (2 fewer scams caught, 4 more false alarms out of 744). Its higher macro-F1 is also within run-to-run noise: the deployed configuration scored 0.862 on CPU but 0.850 when retrained on GPU, and choosing the best of six noisy runs inflates the winner's score. Sweep record: `reports/metrics/distilbert_sweep_colab_20261124T094958Z.json`.

**URLs: the effect of the shortcut fix**

| Feature view | Overall F1 | Hannousse F1 | Hannousse benign flagged |
|---|---|---|---|
| Full URL (shortcut) | 0.983 | 0.773 | 56.7 % |
| Host only | 0.795 | 0.543 | 58.8 % |
| **Mixed (deployed)** | **0.816** | **0.880** | **12.1 %** |

### 7.2 Test set (one-time, frozen models)
Run once on 2026-11-24 with `make eval`, after the sweep decision. The models, configs and splits are fingerprinted (SHA-256) in `reports/metrics/frozen_models.json`, and the results are in `reports/metrics/eval_test_20261124T173748Z.json`.

**Messages (n = 882: 744 ham / 71 spam / 67 scam)**

| Test | Baseline | DistilBERT (deployed) |
|---|---|---|
| Macro-F1 | 0.904 | **0.905** |
| Scam precision / recall | 0.91 / 0.85 | 0.89 / **0.87** |
| Spam precision / recall | 0.86 / 0.83 | 0.86 / 0.83 |
| **Scam flagged as malicious** | 97.0 % (2 scams read as ham) | **100 %** (67 / 67) |
| Spam flagged as malicious | 90.1 % | **91.5 %** |
| Ham false-positive rate | 0.40 % (3 / 744) | **0.27 %** (2 / 744) |

DistilBERT confusion matrix (rows = true, columns = predicted; ham / spam / scam): `[[742, 1, 1], [6, 59, 6], [0, 9, 58]]`.

Test scores are higher than validation (macro-F1 0.905 vs 0.862). With about 70 spam and 70 scam messages per split, a few messages move these numbers. What holds on both splits: **DistilBERT never reads a scam message as normal**, while the baseline missed 2 on test.

**URLs (Random Forest, mixed view, threshold 0.5)**

| Test slice | Precision | Recall | F1 | False-positive rate | ROC-AUC |
|---|---|---|---|---|---|
| Hannousse, full URLs (n = 2,073) | 0.89 | 0.91 | **0.90** | 16.3 % | 0.96 |
| PhiUSIIL, host only (n = 29,709) | 0.90 | 0.56 | 0.69 | 3.6 % | 0.84 |
| All (n = 31,782) | 0.90 | 0.60 | 0.72 | 4.1 % | 0.85 |

On full URLs (the input the app actually receives) the model holds up: F1 0.90 vs 0.88 on validation, though more benign links are flagged (16 % vs 12 %). **PhiUSIIL recall fell from 0.72 to 0.56.** The missed phishing clusters on link shorteners, link-in-bio pages and form builders that never appear in training, a consequence of the strict domain-grouped split: `urlz.fr` (32), `forms.gle` (30), `linkr.bio` (28), `pxlme.me` (23), `zpr.io` (18), `bitly.ws` (15), `apptuts.bio` (15). Seen only as host names, these look like ordinary services. This is the clearest evidence that **phishing on unseen hosting and redirect services is the URL model's main weakness**. Several of these services aren't on the app's shortener-rule list. They are deliberately **not** being added now, because changing the rule after seeing the test set would contaminate it. This is listed as future work.

**Risk bands on the URL test set** (thresholds from validation): of URLs rated **High**, 95.5 % were phishing (target 95 %). 63 % of phishing URLs were rated Medium or High, and 6.3 % of benign URLs.

### 7.3 Challenge set (independent, hand-labelled)
**[TBD]**: n, Cohen's κ between annotators, and metrics. The URL candidates (45 OpenPhish phishing, 45 real benign deep links) are ready for labelling. Messages still need to be collected.

### 7.4 Speed
About 6 s to load the models once (cached), then **0.35–1.0 s per analysis** on a laptop CPU (target: under 3 s).

### 7.5 Conversation mode (synthetic, validation only)
Thresholds for conversation mode come from synthetic chats of 4–16 validation messages (`make conversation-thresholds`, `reports/metrics/conversation_thresholds.json`): 500 benign chats, 300 with one spam/scam message inserted at a random position, and 300 with a spam/scam message **split at a sentence boundary across two consecutive turns**. Medium is set so ≤ 3 % of benign chats are flagged and High so ≤ 0.5 %.

| Window size | Benign chats flagged (Medium / High) | Inserted caught (Medium / High) | Split caught (Medium / High) |
|---|---|---|---|
| 1 message | 2.0 % / 0.0 % | 95.3 % / 93.0 % | 95.7 % / 91.0 % |
| up to 3 messages (used) | 2.8 % / 0.4 % | 95.3 % / 93.0 % | 95.7 % / **95.7 %** |

Windows recover a scam spread over two messages at the High band (91.0 % → 95.7 %) at a small false-alarm cost. The thresholds are much higher than for a single message (Medium ≥ 0.994 vs 0.003), because a few benign validation messages that DistilBERT confidently misreads appear in many synthetic chats. The cross-message rules fired on **0 %** of benign synthetic chats; they also never fire on the inserted or split chats, which contain no conversation patterns, so their benefit is shown only by the worked examples in the app. **Limitation:** synthetic chats are unrelated messages stitched together, not real conversations; real conversation data is future work.

## 8. Error analysis
Full details are in `reports/error_analysis.md`.
1. **DistilBERT never reads a scam message as normal**, on validation or test. The baseline missed 2 scams on test.
2. **Spam/scam label noise.** Premium-rate SMS are labelled inconsistently ("free entry" 9 scam / 7 spam; "150p" 64 / 79), which accounts for most message errors.
3. **Transactional false alarms.** Delivery, top-up and bank notices share vocabulary with scams. This is the main user-facing weakness.
4. **Phishing on ordinary-looking domains and on unseen redirect/hosting services** (link-in-bio pages, form builders, little-known shorteners) is missed by a text-only URL model. On test, PhiUSIIL recall fell from 0.72 to 0.56 for this reason.
5. **Shortener blind spot.** Bare `tinyurl.com` scored 0.45; fixed in the app with a rule.
6. **Explanation limit.** Leave-one-out finds no single word in messages packed with scam cues. The app says so explicitly.

## 9. Limitations and ethics
- English only. The training data are older UK/India SMS and 2020–2024 URLs, so new scam styles may be missed.
- Scores are uncalibrated. Only bands are shown.
- False reassurance is the biggest risk: every Low result states it doesn't guarantee safety, and the advice points to official channels.
- Conversation mode is measured only on synthetic chats, and the model still reads at most 3 messages at once; clues further apart rely on hand-written rules, which can't anticipate new scripts. A model trained on real conversations, or a long-context model, is future work.
- Context rules are hand-written and unevaluated. They can't help when the user doesn't know the context, and a wrong answer (e.g. "unknown number" for a real bank short code) can over-warn. They never lower a rating.
- Privacy: no storage, no logging, no link visits, telemetry off. Context answers aren't stored either.
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
| Precision, recall, F1, confusion matrices, FPR | Done (validation and one-time test) |
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
