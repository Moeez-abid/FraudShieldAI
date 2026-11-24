# Week 3 Summary

Week 3 connects everything: URL extraction, a single `analyse()` pipeline, frozen-model evaluation and the Streamlit app.

## 1. URL extraction (`inference/extract_urls.py`)
- Finds normal links, `www.` links, bare domains (`bit.ly/abc`, `hbl-secure.top/login`), raw IPs with ports, and **defanged** links (`hxxp://`, `[.]`, `(.)`, `[dot]`), which it restores before analysis.
- Skips common false positives: `Rs.500`, `e.g.`, `report.pdf`, version numbers, e-mail addresses and `Hi.How`. A bare host with no path or `www.` must use a common TLD or a 2-letter country code.
- Strips trailing punctuation but keeps balanced brackets (`…/Bracket_(disambiguation)`). Removes duplicates and caps the list at 5 links.
- Links pasted without `http(s)://` get `http://` added for scoring, and "doesn't use HTTPS" is then hidden from the reasons, because the user just didn't type a scheme.

## 2. Pipeline (`inference/pipeline.py`)
`FraudShield.load().analyse(text)` returns a `Report`:
- **overall**: the highest band among the message and its links.
- **headline**: worded by band ("looks like a scam" for High, "may be a scam" for Medium).
- **advice**: different for Low, Medium and High.
- **message**: band, predicted class, top words (occlusion), `distributed` flag, rule-based signals, and `capped` if spam was limited to Medium.
- **urls**: band, score, SHAP reasons, and `defanged` / `scheme_given` flags for each link.
- **timings_ms** for each step.

Rules:
- Input is normalised and truncated to 2,000 characters.
- Input that is only a link (at most 2 other words) skips the message model.
- **Spam is capped at Medium** (`spam_max_band` in `configs/message.yaml`), so ordinary promotions are never shown as High danger. Scam predictions are not capped.
- A message the model reads as `ham` is never rated High.
- `load()` uses DistilBERT and falls back to the baseline if torch isn't installed. It refuses to start if `risk_thresholds.json` was computed for a different message model.
- Nothing is logged or stored, and URLs are never fetched.

**Fixed while integrating:**
- The numeric feature `tld_len` was being read as the one-hot TLD ".len" in URL reasons. Features are now matched by transformer prefix.
- "Includes http(s)://" was dropped as a reason, because every scored URL has a scheme.

## 3. Frozen-model evaluation (`evaluation/evaluate.py`)
- `make eval-val` and `make eval` score the **saved** models. Previously `make eval` retrained DistilBERT.
- Also scores the challenge set once `label_final` is filled in. Challenge URLs are scored the way the app scores them, with a scheme added.
- Writes `reports/metrics/eval_<split>_<time>.json`, confusion matrices, and per-example error CSVs to `reports/errors/` (git-ignored because they contain dataset text).
- Validation numbers match the training-time numbers exactly:

| Model | Validation result |
|---|---|
| baseline | macro-F1 0.865, scam recall 0.82 |
| DistilBERT | macro-F1 0.862, scam recall 0.90 |
| URL RF (mixed) | F1 0.816 overall; Hannousse 0.880; PhiUSIIL 0.813 |

- While building this I found that `has_scheme` exactly separates the two URL sources (PhiUSIIL never has one, Hannousse always does). The model barely uses it (importance 0.013, 14th of 15 features), and bare domains score sensibly (`google.com` 0.37, `hbl.com` 0.33, `daraz.pk` 0.17, all Low). The evaluator scores dataset rows on the exact strings used in training, so the numbers stay comparable.

## 4. Streamlit app (`app/streamlit_app.py`, `make app`)
- Six one-click examples: bank scam, prize scam, delivery notice, promotion, normal chat and a bare link.
- A verdict banner, then two columns. **Message**: band, model reading, highlighted words, a note when no single word decides, and rule-based signals labelled as separate from the model. **Links**: one card per URL with band, the restored URL as plain text, notes on disguised or scheme-less links, and ▲/▼ reasons.
- The advice box changes with the verdict. A footer shows the analysis time and the model used.
- The sidebar covers privacy, how it works, limitations and validation metrics.
- **Safety:** all user text and URLs are HTML-escaped and rendered with `st.html` (no markdown auto-linking), so nothing in the output is clickable. Streamlit usage statistics are turned off in `.streamlit/config.toml`.

**Checked in the browser** (M1 laptop, CPU inference):

| Example | Overall | Notes |
|---|---|---|
| Bank scam (defanged link) | High | Message scam/High; link restored → High (`.top`, phishing words, hyphens); 4 rule-based signals |
| Prize scam | High | "No single word decides" note; prize and payment signals |
| Delivery notice | Medium | Known false positive (read as scam) |
| Promotion + `outfitters.com.pk` | Medium | Spam capped at Medium; link Low |
| Normal chat | Low | — |
| `paypal.com-secure-login.xyz/verify?…` | High | Message model skipped; `.xyz`, randomness, 3 phishing words |

**Speed:** about 6 s to load the models once (cached), then **0.35–1.0 s per analysis**, well under the 3 s target. Message occlusion takes about 0.05–0.5 s and SHAP about 0.45 s per link.

## 5. Tests
61 tests, run with `make test`, cover URL extraction (21), the pipeline end to end with stand-in models (8), explainers, risk bands, features, data handling and metrics. **No trained model files are needed to run them.**

## 6. Open items for Week 4
- **Challenge set:** collect and double-label it (`data/challenge/`), including benign deep links and normal delivery and bank messages. Then run `make eval-val` to see challenge results.
- **DistilBERT sweep on Colab** (optional), then freeze the models, run `make thresholds`, then run **`make eval` once** on test.
- **Survey and user trials** (`docs/survey.md`) using `make app`.
- **Error analysis** from `reports/errors/`, and the final report and demo.
- **Known weak spots to discuss:** delivery and bank-style notices flagged as Medium; leave-one-out explanations on redundant scam text; English only.
