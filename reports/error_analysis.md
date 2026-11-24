# Error Analysis (validation)

Source: `make eval-val` / `make eval` → `reports/errors/*/errors_*.csv` (git-ignored; regenerate locally). Sections 1–3 analyse **validation** errors, inspected before the test evaluation. Section 4 covers the **one-time test evaluation**. No changes were made after it.

## 1. Messages

| | Baseline (TF-IDF + LR) | DistilBERT |
|---|---|---|
| Errors (of 881) | 34 | 33 |
| scam → read as ham (missed) | **0** | **0** |
| scam → spam | 12 | 7 |
| spam → scam | 7 | 14 |
| spam → ham (missed) | 11 | 8 |
| ham → spam / scam (false alarm) | 4 | 4 |

Neither model reads any scam message as normal: every scam gets some warning. All other errors fall into four groups.

### A. Spam/scam boundary: about 21 of DistilBERT's 33 errors
Premium-rate SMS ("FreeMsg… 150p/msg", "FREE entry into our £250 weekly comp", "Had your mobile 11mths+?", dating lures) sit on the line between promotion and deception. **The dataset labels them inconsistently**:

| Phrase in message | Labelled scam | Labelled spam |
|---|---|---|
| "free entry" | 9 | 7 |
| "150p" | 64 | 79 |
| "11 mths" | 5 | 5 |
| "camera phone" | 8 | 7 |
| "sms.ac" | 4 | 4 |

Near-identical messages carry different labels (e.g. "FREE entry … 88877" is spam, the "… 80086" variant is also spam, while similar comps are scam). Exact-class accuracy on these is limited by label noise, not just by the models. That's why the report shows **"flagged as malicious"** (spam or scam) next to exact-class recall, and why the app treats spam and scam alike for everything except the Medium cap.

### B. Probable mislabels in the source data: about 5 errors
Messages labelled spam that read as ordinary jokes or chat, which the models call ham:
- "How come it takes so little time for a child who is afraid of the dark to become a teenager…"
- "Do you ever notice that when you're driving, anyone going slower than you is an idiot…"
- "For sale - arsenal dartboard. Good condition…"
- horoscope messages ("ROMCA Everyone around should be responding well…")

These count as errors but arguably aren't. They're left as labelled to avoid tuning labels toward the model.

### C. Legitimate transactional messages flagged: false alarms
- "Your account has been refilled successfully by INR 100.00… Transaction ID…" → read as scam
- Short technical messages ("Host-based IDPS for linux systems.", "Is toshiba portege m100 gd?") → read as spam
- Outside the dataset (app testing): "Your Daraz order #48213 has been shipped…" → Medium, read as scam

Transactional notices share vocabulary with scams (account, order, transaction, shipped, amounts, IDs), and the training data has few legitimate ones. **This is the main user-facing weakness**, and the challenge set should include real delivery and bank notices.

### D. Missed spam written like chat: about 3 errors
- "Money i have won wining number 946 wot do i do next" → ham
- "Hello. We need some posh birds and chaps to user trial prods for champneys… I need your address and dob asap." → ham. This one is arguably data harvesting, i.e. a scam.

Spam without promotional markers (no "free", prices or short codes) looks like normal chat.

## 2. URLs (mixed model, threshold 0.5)

5,377 errors in 36,498 validation URLs:

| Source | Missed phishing (FN) | Benign flagged (FP) |
|---|---|---|
| PhiUSIIL (host only) | 4,489 | 713 |
| Hannousse (full URL) | 88 | 87 |

### A. Phishing on ordinary-looking domains: most of PhiUSIIL's FN
Examples: `gun-stores.net`, `mobilecarhandel.pl`, `labinstruments.org`, `dossier-amendes.fr`. These are often compromised or freshly registered sites whose host names look normal. **A host-name model can't catch them.** Catching them needs the path (only available for Hannousse rows) or outside signals such as domain age, reputation and blocklists, all out of scope because the app never contacts external services.

### B. URL shorteners: 135 of 140 `tinyurl.com` phishing URLs missed
PhiUSIIL stores them as the bare host `tinyurl.com`, which scores 0.45. The trained `is_shortener` feature carries little weight (importance 0.006): the model learned "short host name" instead. Full shortener links like the ones users paste score higher (`tinyurl.com/4abc9x` 0.71, `bit.ly/3xYz` 0.97), but not reliably.
**Fix applied (rule, outside the model):** any link on a known shortener is rated at least **Medium**, with the reason "link shortener hides the real destination". The trained model is unchanged.

### C. Legitimate sites with unusual host names: most FP
Examples: `h2020.net` (0.95), `kx935.com` (0.90), `oxford.anglican.org`, `bywgraffiadur.cymru`, `iz.sk`, and `.mil` domains (44 FP). Digits, rare TLDs and extra subdomains are phishing signals in the training data, so legitimate sites with them get flagged. Rare-but-legitimate TLDs (.mil, .cymru) are effectively unseen.

### D. Hosting platforms: handled by the split, still a risk
Thousands of phishing pages live on `web.app`, `firebaseapp.com`, `repl.co`, `github.io` and similar. Grouping by registered domain keeps each platform in a single split, so results reflect platforms never seen in training. In practice, a legitimate `*.web.app` site may be flagged and a phishing page on a platform the model knows may score lower. Only 12 FN were on these platforms in validation.

## 3. Consequences for the report and app
1. Report **flagged-as-malicious recall** alongside exact-class metrics, and explain the spam/scam label noise (1A).
2. List **transactional false alarms** (1C) and **ordinary-domain phishing** (2A) as the main limitations in the app sidebar and the report.
3. Shortener rule added (2B).
4. Challenge set priorities: real delivery and bank notices (ham), current smishing, benign deep links, phishing on ordinary-looking domains, and shortener links.

## 4. Test set (one-time evaluation, frozen models)

**Messages, DistilBERT: 23 errors in 882**

| true → predicted | count |
|---|---|
| scam → spam | 9 |
| spam → scam | 6 |
| spam → ham | 6 |
| ham → spam / scam | 2 |
| scam → ham | **0** |

The same patterns as validation:
- **Spam/scam boundary (15 errors).** Premium-rate dating lines, "FREE … Txt … to 8xxxx" competitions, an advance-fee message ("7.3 milion Dollar to transfer…") read as spam, and a Pokémon-coins phishing link read as spam. All are still flagged as malicious.
- **Probable mislabels (about 3).** Horoscope and joke messages labelled spam ("ROMCAPspam Everyone around should be responding well…", "Did you hear about the new 'Divorce Barbie'?").
- **Missed spam that reads as chat (about 3).** "lam Frederick banner. Pls conatct me asap…", "RCT' THNQ Adrian for U text.", "thesmszone.com lets you send free anonymous… messages".
- **False alarms (2).** A contact-list notification from a free-SMS site (arguably spam itself), and a chat message containing a membership number.

The baseline additionally misread 2 scams as ham.

**URLs: 4,700 missed phishing URLs of 11,797**
Concentrated on services never seen in training (the domain-grouped split keeps each service in one split): `onlinequranteaching.com` (51), `urlz.fr` (32), `forms.gle` (30), `linkr.bio` (28), `pxlme.me` (23), `couponsonfire.com` (18), `zpr.io` (18), `bitly.ws` (15), `apptuts.bio` (15). These are mostly redirect, link-in-bio and form services, plus compromised ordinary sites. This confirms section 2A at a larger scale. The shortener rule wasn't extended to these services after seeing the test set (that would contaminate it). It's listed as future work.
