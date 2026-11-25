# Model Cards

Three models power FraudShield AI. Numbers are on the **validation** split unless marked **test**. The test results come from the one-time evaluation of the frozen models (fingerprints in `reports/metrics/frozen_models.json`).

---

## 1. DistilBERT message classifier (primary)

| | |
|---|---|
| **File** | `models/distilbert/` (256 MB; not in git, see README "Sharing") |
| **Base model** | `distilbert-base-uncased` (Sanh et al., 2019), fine-tuned |
| **Task** | 3-class: `ham` (normal), `spam` (promotion, no deception), `scam` (deceptive: money, credentials, malicious link) |
| **Training data** | UCI SMS Spam Collection + Mendeley SMS Phishing Dataset, merged and deduplicated: 5,875 messages (4,959 ham / 469 spam / 447 scam); stratified 70/15/15 split |
| **Training** | Class-weighted cross-entropy, lr 3e-5, max length 128, effective batch 16, 4 epochs, best checkpoint by validation macro-F1. CPU, about 45 min |
| **Script / config** | `src/fraudshield/models/distilbert.py`, `configs/message.yaml` |
| **Sweep** | 6 runs on Colab T4 (lr × max length). The best run (lr 5e-5 / 128, val macro-F1 0.882) was **not adopted**: lower scam recall (0.866) and double the ham FPR (1.08 %) under the pre-set selection rule. See `reports/final_report.md` §7.1 |

**Validation results (n = 881: 744 ham / 70 spam / 67 scam)**

| Metric | Value |
|---|---|
| Macro-F1 | 0.862 |
| Scam: precision / recall / F1 | 0.80 / 0.90 / 0.85 |
| Spam: precision / recall / F1 | 0.83 / 0.69 / 0.75 |
| Ham: precision / recall / F1 | 0.99 / 0.995 / 0.99 |
| Scam flagged as malicious (spam or scam) | 100 % |
| Ham flagged as malicious (false-positive rate) | 0.54 % |

**Test results (n = 882):** macro-F1 **0.905**; scam P / R 0.89 / 0.87; spam P / R 0.86 / 0.83; **scam flagged as malicious 100 %**; ham false-positive rate **0.27 %**.

**Risk bands** (malicious score = 1 − P(ham); tuned on validation): Medium ≥ 0.003 (3 % of ham flagged), High ≥ 0.988 (precision 0.98, recall 0.93). Spam predictions are capped at Medium in the app.

**Intended use.** A second opinion on English SMS, chat or email text, shown to people as Low / Medium / High with reasons. Not for automated blocking.

**Limitations**
- Training messages date from 2011–2022 and are largely UK/India SMS. Current local scam styles (e.g. Pakistani wallets and couriers) are under-represented.
- The spam/scam boundary in the data is inconsistent for premium-rate SMS (see `reports/error_analysis.md`), which limits exact-class accuracy.
- Legitimate transactional messages (deliveries, top-ups, bank alerts) can be flagged.
- Scores are uncalibrated and near 0 or 1. Show bands, not percentages.
- English only. Roman Urdu and code-mixed text are not supported.

**Explanations.** Word occlusion on the malicious log-odds (`explain/text_occlusion.py`). If no single word decides the verdict, the app says so.

---

## 2. TF-IDF + Logistic Regression (baseline / fallback)

| | |
|---|---|
| **File** | `models/baseline_tfidf_lr.joblib` (2 MB) |
| **Features** | Word 1–2-grams + character 3–5-grams (TF-IDF, sublinear) |
| **Classifier** | Logistic Regression, `class_weight=balanced`, C = 0.3 (tuned on validation) |
| **Data / split** | Same as DistilBERT |

**Validation:** macro-F1 0.865; scam P / R 0.85 / 0.82; spam P / R 0.79 / 0.76; ham false-positive rate 0.54 %; 100 % of scam flagged.

**Test:** macro-F1 0.904; scam recall 0.85; scam flagged 97 % (**2 scams read as ham**); ham false-positive rate 0.40 %.

**Use.** The comparison baseline required by the proposal, and the app's fallback when torch isn't installed (run `make thresholds` after switching). It's statistically tied with DistilBERT on macro-F1 but has lower scam recall (0.82 vs 0.90).

---

## 3. URL Random Forest (mixed view)

| | |
|---|---|
| **File** | `models/url_rf_mixed.joblib` (45 MB) |
| **Features** | 31 lexical/structural features from the URL text (lengths, character counts, subdomains, IP host, punycode, brand outside domain, phishing keywords, entropy, digit runs, shortener, TLD one-hot). **No network access**: the page is never fetched (`features/url_features.py`) |
| **Classifier** | Random Forest, 300 trees, max depth 20, `class_weight=balanced` |
| **Training data** | PhiUSIIL (2024) **host names only, leading `www.` removed**, plus Hannousse & Yahiouche (2020) **full URLs**: 246,715 URLs |
| **Split** | StratifiedGroupKFold by registered domain (no domain or hosting platform in two splits): 178k / 36k / 32k |

**Why the mixed view.** In PhiUSIIL, every benign URL is an `https://www.` homepage. Trained on raw URLs, a model learns "path / http / no www → phishing" and flags 57 % of real benign deep links. The mixed view removes these shortcuts (`reports/week2_summary.md`).

**Validation results**

| Slice | Precision | Recall | F1 | False-positive rate |
|---|---|---|---|---|
| All (n = 36,498) | 0.94 | 0.72 | 0.82 | 4.0 % |
| Hannousse, full URLs (n = 1,452) | 0.88 | 0.88 | 0.88 | 12.1 % |
| PhiUSIIL, host only (n = 35,046) | 0.94 | 0.72 | 0.81 | 3.7 % |

**Test results**

| Slice | Precision | Recall | F1 | False-positive rate |
|---|---|---|---|---|
| Hannousse, full URLs (n = 2,073) | 0.89 | 0.91 | 0.90 | 16.3 % |
| PhiUSIIL, host only (n = 29,709) | 0.90 | 0.56 | 0.69 | 3.6 % |

PhiUSIIL recall drops from validation (0.72 → 0.56) because the missed phishing sits on redirect, link-in-bio and form services absent from training (`urlz.fr`, `forms.gle`, `linkr.bio`…). At the High band, 95.5 % of test URLs rated High were phishing.

**Risk bands** (tuned on the Hannousse slice, the only full-URL data): Medium ≥ 0.438 (15 % of benign flagged), High ≥ 0.663 (precision 0.95, recall 0.73). App rule: link shorteners are rated at least Medium.

**Limitations**
- Phishing on ordinary-looking or compromised domains is often missed. A text-only check can't see domain age, reputation or page content.
- Legitimate sites with digits, rare TLDs (e.g. `.mil`, `.cymru`) or many subdomains can be flagged.
- Path, query and scheme behaviour is learned from only about 8k Hannousse training URLs from 2020.
- Free hosting platforms (`web.app`, `pages.dev`, `github.io`) host both legitimate and phishing sites.

**Explanations.** SHAP TreeExplainer, top 4 contributions written as neutral statements with ▲ / ▼ direction (`explain/url_shap.py`).

---

## Shared ethical notes
- **Privacy:** the app keeps nothing. Inputs aren't logged or stored, links aren't opened, and Streamlit usage statistics are disabled.
- **False reassurance** is the main harm. Every Low result says it doesn't guarantee safety, and the advice always points to official channels.
- **False alarms** on legitimate notices could teach users to ignore warnings. The Medium band and the spam cap are there to limit alarm fatigue.
- **Optional context rules** (`inference/context_rules.py`) sit outside all three models. The user's answers about the sender can only raise the overall band, never lower it, and never change model inputs. With no context, results match the cards above exactly. The rules are not evaluated on labelled data.
- **Conversation mode** reuses the frozen DistilBERT on overlapping windows of up to 3 messages (≤ 128 tokens) and takes the maximum; messages over 128 tokens are read in overlapping chunks. It has its own thresholds (Medium ≥ 0.994, High ≥ 0.999), tuned on synthetic chats of validation messages so ≤ 3 % / 0.5 % of benign chats are flagged (`reports/metrics/conversation_thresholds.json`, report §7.5). Cross-message rules only raise the rating. The model was never trained on conversations.
- All datasets are CC BY 4.0 and cited in the README.
