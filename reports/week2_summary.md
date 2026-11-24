# Week 2 Summary

All numbers are on the **validation** split. The test sets are still untouched and reserved for Week 4.

## 1. URL data: fixing the PhiUSIIL shortcuts

**Problem found in Week 1.** In PhiUSIIL, 100 % of benign URLs are `https://www.<domain>` homepages with no path. Among phishing URLs, 49 % use https, 41 % start with `www.` and 28 % have a path. A model trained on full URLs learned "path / http / no www → phishing".

**Fix.**
1. Added the **Hannousse & Yahiouche (2020)** dataset (Mendeley, CC BY 4.0, 11,430 URLs, balanced). In it, both classes have deep links and both schemes:

   | Hannousse | has path | https | starts with www. |
   |---|---|---|---|
   | benign | 62 % | 44 % | 67 % |
   | phishing | 82 % | 33 % | 21 % |

2. **Mixed feature view** (`clean_urls.py`):
   - PhiUSIIL rows contribute **only the host name, with a leading `www.` removed**, for both classes. Its scheme, path and www artefacts can no longer be learned.
   - Hannousse rows contribute **full URLs**, so path, query and scheme features are learned only from data where both classes have them.
3. **Split:** 246,715 URLs, grouped by registered domain with `StratifiedGroupKFold` on (source × label): 178k train / 36k val / 32k test. Hosting platforms such as `web.app`, `firebaseapp.com` and `repl.co` (thousands of phishing sites each) stay wholly inside one split. That's strict, because the model is tested on platforms it never saw, but it makes the class ratio differ between splits (val 45 % phishing, test 37 %).
4. The feature cache now checks the exact source strings, so a changed `feature_url` can't silently reuse stale features.

**Results (validation, Random Forest).** The Hannousse slice is the only full-URL data, so it's the realistic one.

| Feature view | Overall F1 | Hannousse F1 | Hannousse benign flagged | Top features |
|---|---|---|---|---|
| full URL (shortcut) | 0.983 | 0.773 | **56.7 %** | `is_https`, `path_len` |
| host only | 0.795 | 0.543 | 58.8 % | `n_subdomains`, `host_entropy` |
| **mixed (serving)** | 0.816 | **0.880** | **12.1 %** | `n_subdomains`, `n_dots`, `host_entropy` |

The full-URL model's 0.98 is inflated: on real deep links it raises false alarms on 57 % of benign URLs. The mixed model is the one the app uses.

**Sanity probes** (phishing score from the mixed model, band using the Section 4 thresholds):

| URL | Expected | Score | Band |
|---|---|---|---|
| `https://www.amazon.com/dp/B08N5WRWNW` | benign | 0.35 | Low |
| `https://en.wikipedia.org/wiki/Phishing` | benign | 0.04 | Low |
| `https://github.com/huggingface/transformers/issues` | benign | 0.41 | Low |
| `http://www.bbc.co.uk/news/technology` | benign | 0.37 | Low |
| `https://www.daraz.pk/products/wireless-earbuds-i123456789.html` | benign | 0.55 | Medium |
| `https://www.paypal.com/signin` | benign | 0.52 | Medium |
| `http://paypal.com-secure-login.xyz/verify` | phishing | 0.87 | High |
| `http://192.168.4.20/bank/login.php` | phishing | 0.99 | High |
| `https://secure-hbl-verify.top/account/update?id=88123` | phishing | 0.82 | High |
| `http://appleid.apple.com.verify-user.info/login` | phishing | 0.91 | High |

Ten hand-written URLs are only a smoke test. The challenge set must include benign deep links.

## 2. DistilBERT message classifier

`distilbert-base-uncased`, 3 classes, class-weighted cross-entropy, lr 3e-5, max length 128, effective batch 16 (8 × 2 gradient accumulation), early stopping on val macro-F1, 4 epochs. Trained on **CPU**: Apple MPS ran out of memory twice on the 8 GB M1. Use Colab for the hyperparameter sweep (`notebooks/train_distilbert_colab.ipynb`).

| Validation | Baseline (TF-IDF + LR) | DistilBERT |
|---|---|---|
| Macro-F1 | **0.865** | 0.862 |
| ham FPR | 0.54 % | 0.54 % |
| spam P / R / F1 | 0.79 / 0.76 / 0.77 | 0.83 / 0.69 / 0.75 |
| scam P / R / F1 | 0.85 / 0.82 / 0.83 | 0.80 / **0.90** / **0.85** |
| scam flagged as malicious | 100 % | 100 % |
| spam flagged as malicious | 82.9 % | **88.6 %** |

- The two models are statistically indistinguishable on macro-F1. The validation set has only 70 spam and 67 scam messages, so a difference of ±0.01 is noise.
- DistilBERT has **higher scam recall** (0.90 vs 0.82) and flags more spam. Phishing recall is the proposal's priority, so **DistilBERT is the primary message model** and the baseline is the fallback.
- The `--sweep` (lr × max length) hasn't been run yet. Run it on Colab.

## 3. Explanations

| Component | Method | Notes |
|---|---|---|
| Messages, model-based | Word occlusion on the **log-odds** of malicious, from exact log-probabilities | float32 softmax rounds P(ham) to 0 for confident predictions, so log-probs are required. Top words are shown only for spam/scam predictions. |
| Messages, rule-based | Regex tags for urgency, credential request, account threat, prize, payment, impersonation and link prompt (includes Pakistani banks, couriers and wallets) | Shown separately as heuristics |
| URLs | SHAP TreeExplainer, top 4 contributions, written neutrally ("Subdomain levels: 0") with a raises/lowers-risk direction | About 1 s per URL on CPU |

**Known limitation.** Leave-one-out can't attribute redundant evidence. A message with many scam cues stays confidently scam whichever single word is removed. The explainer then sets `distributed: true`, and the app should say that the overall pattern, not one word, drives the verdict.

## 4. Risk bands

Thresholds are chosen on validation (`make thresholds` writes `models/risk_thresholds.json`). **High** is the lowest score with precision ≥ the target. **Medium** is the lowest score with benign-flagged rate ≤ the target. Overall verdict = the highest band of any component.

| Model | Tuned on | Medium ≥ | High ≥ | At Medium (P / R / benign flagged) | At High (P / R) |
|---|---|---|---|---|---|
| DistilBERT (malicious = 1 − P(ham)) | all val messages | 0.003 | 0.988 | 0.86 / 0.97 / 3.0 % | 0.98 / 0.93 |
| URL RF mixed | Hannousse val slice (n = 1,452) | 0.438 | 0.663 | 0.86 / 0.90 / 15 % | 0.95 / 0.73 |

DistilBERT's scores are near 0 or 1, which is why its thresholds look extreme. **Neither model is calibrated**, so the app must show bands, not percentages.

## 5. Failure cases found so far

1. **Legitimate delivery notice → Medium/spam.** `Your Daraz order #48213 has been shipped…` scores 0.77, driven by `#48213`, `order` and `shipped`. Delivery scams in the training data make all delivery messages look suspicious.
2. **Legitimate promotions → High/spam.** `50% off all shoes this weekend…` is correctly spam, but "High" overstates the danger of normal marketing. Option: let spam-only predictions top out at Medium.
3. **Legitimate deep links on less common sites → Medium** (daraz.pk, paypal.com/signin).
4. **Leave-one-out explanation is empty** for heavily redundant scam messages (see Section 3).

## 6. Open items for Week 3 and later
- Run the DistilBERT sweep on Colab; record the run in `reports/metrics`.
- Decide whether spam-only predictions should be capped at Medium.
- Add `evaluate.py` so `make eval` scores the frozen saved models instead of retraining.
- Build the Streamlit app on `message_model`, `text_occlusion`, `text_signals`, `url_shap` and `risk`.
- Collect the challenge set, including benign deep links and normal delivery and bank messages.
