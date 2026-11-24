# Week 1 Summary

All numbers are on the **validation** split. The test sets haven't been used yet.

## Data
| Task | Rows | Classes | Split |
|---|---|---|---|
| Messages (UCI + Mendeley, deduplicated) | 5,875 | 4,959 ham / 469 spam / 447 scam | stratified 70/15/15 |
| URLs (PhiUSIIL, URL string only) | 235,335 | 134,850 benign / 100,485 phishing | grouped by registered domain (175,506 domains) |

- The Mendeley set contains most of the UCI messages: 5,670 duplicates were removed. For 265 messages UCI said "spam" and Mendeley said "smishing", and those are labelled `scam`.
- PhiUSIIL label encoding: 1 = legitimate and 0 = phishing (checked against the paper's class counts).

## Message baseline (TF-IDF word + char n-grams, Logistic Regression, C=0.3)
| Class | Precision | Recall | F1 |
|---|---|---|---|
| ham | 0.98 | 0.99+ | 0.99 |
| spam | 0.79 | 0.76 | 0.77 |
| scam | 0.85 | 0.82 | 0.83 |

- Macro-F1 is **0.865**, and **0.5 %** of ham messages are wrongly flagged.
- Scam vs ham: **100 %** of scam messages are flagged as malicious, and 82 % get the exact `scam` label (the rest are called spam).
- Most errors are spam/scam confusion, which is the line DistilBERT needs to improve in Week 2.

## URL Random Forest
| View | Precision | Recall | F1 | FPR | ROC-AUC | Random-split F1 |
|---|---|---|---|---|---|---|
| Full URL | 0.9997 | 0.980 | **0.990** | 0.0003 | 0.999 | 0.996 |
| Host only | 0.933 | 0.793 | **0.857** | 0.052 | 0.920 | 0.830 |

### Dataset artefact (important)
In PhiUSIIL, **100 % of benign URLs are https homepages with no path**. Among phishing URLs, 49 % use https and 28 % have a path. The full-URL model's top features are `is_https` and `path_len`, so its 0.99 F1 mostly comes from learning that shortcut.

Probe with real benign deep links (phishing score from the full model → host-only model):

| URL | Full | Host |
|---|---|---|
| `https://www.amazon.com/dp/B08N5WRWNW` | 0.99 | 0.24 |
| `https://en.wikipedia.org/wiki/Phishing` | 0.98 | 0.14 |
| `http://www.bbc.co.uk` | 0.90 | 0.02 |
| `http://paypal.com-secure-login.xyz/verify` | 1.00 | 0.95 |
| `http://192.168.4.20/bank/login.php` | 1.00 | 1.00 |

The full-URL model flags almost any normal link with a path as phishing, so it isn't usable in the app as it stands. The host-only model is less accurate on paper but behaves sensibly.

The domain-grouped split also changes results: grouping lowers full-view F1 from 0.996 to 0.990, as expected when leakage is removed. For the host view the random split happens to score lower; that comparison needs a look in Week 2.

### Options for Week 2
1. **Add benign deep-link URLs** (with paths, queries and http) from a second source, retrain the full-URL model and check that `is_https` / `path_len` stop dominating. Recommended.
2. Use the **host-only model** in the app and describe path analysis as out of scope.
3. Combine the two: a host model for the verdict plus rule-based path signals (`login`, `.php`, `@`, `//`) shown as heuristics.

Whichever option is chosen, the challenge set must include benign deep links so this failure shows up in the final evaluation.
