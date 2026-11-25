# Demo Script (about 8 minutes)

**Before the demo**
- Run `make app` at least 2 minutes early. The first analysis loads the models (about 6 s), so click one example once to warm it up.
- Have `reports/week2_summary.md` (the shortcut table) and `reports/figures/` open in another tab.
- Close other heavy apps. DistilBERT runs on CPU, and an 8 GB laptop is fine for inference.
- Backup: if the app fails, run `make test` and walk through `reports/final_report.md`.

Speakers: **M** = Moeez (messages), **I** = Ijtaba (URLs).

---

### 1. Problem (M, 45 s)
"Phishing messages impersonate banks, couriers and prize draws. APWG counted about 3.8 million phishing attacks in 2025. Spam filters only work inside one app. FraudShield is a manual second opinion that works for anything you can copy and paste, and it tells you **why**."

### 2. Live: a bank scam (M, 1 min 30 s)
Click **Bank scam**.
- Point at the **High risk** banner and its one-line summary.
- **Message column:** the model reads it as *Scam / phishing*. Here no single word decides; the whole message matches scam patterns. Point out that the app says this instead of highlighting random words.
- **Scam signals:** urgency ("within 24 hours"), credential request ("verify your account"), account threat, impersonation (HBL). Say these are **rule-based checks, shown separately from the model**.
- **Link column:** the message hid the link as `hxxp://hbl-secure-verify[.]top`. We restore it and score it as **High** (`.top` TLD, phishing keywords, hyphens). The link is shown as plain text and is **never opened or clickable**.

### 3. Live: things that should *not* alarm (M, 1 min)
- **Normal chat** → Low. "No strong warning signs", and it still says Low isn't a guarantee.
- **Promotion** → Medium, *capped*. "Spam is annoying but not deceptive, so we never show it as High danger. The Outfitters link itself is Low."

### 4. Live: just a link (I, 1 min)
Click **Just a link** (`paypal.com-secure-login.xyz/verify…`).
- Only a link, so the message model is skipped.
- Reasons with ▲: `.xyz`, a random-looking host name, 3 phishing words. "These are SHAP contributions from the Random Forest. The link's text is the only input; the page is never fetched."
- Type `bit.ly/3abcXYZ`. It's rated at least Medium by rule, because a shortener hides the destination.

### 5. The data story (I, 1 min 30 s)
Show the table from `reports/week2_summary.md`:
"The largest public phishing-URL dataset has a trap: every legitimate URL is an `https://www.` homepage. Our first model scored F1 0.98, but on real legitimate deep links it flagged **57 %** of them as phishing. It had learned 'has a path → phishing'. We added a second dataset and trained on host names for the first and full URLs for the second. Real-link false alarms fell to **12 %**. The headline number went down, and the model became usable."

Then: "We also split by **registered domain**, so the model is always tested on domains and hosting platforms it has never seen."

### 6. Honest weak spot (M, 45 s)
Click **Delivery notice** → Medium.
"A real Daraz shipping message gets flagged. Delivery scams in the training data make all delivery messages look suspicious. It's documented in our error analysis, and it's why the challenge set includes real delivery and bank notices."

### 6b. Live: context (M, 45 s)
Click **New number** ("Hi Ammi, it's me… this is my new number") → **Low**: as text it's harmless.
Open **Add context**, choose *An unknown mobile number*, click **Analyse** → **Medium**, with the reason and "Without your context, the models rated this Low".
"Some scams only show themselves through context. Context can only raise the rating, never lower it, because contacts get hacked and sender names get spoofed. It never goes into the models, so our test results still hold."

### 7. Results and engineering (I, 1 min)
- **Test set (run once, frozen models):** DistilBERT macro-F1 **0.905**, **every scam flagged (67/67)**, 0.27 % false alarms on normal messages. The baseline missed 2 scams.
- Links: F1 **0.90** on full URLs. Weak spot: phishing on unseen redirect and link-in-bio services pulled recall on the PhiUSIIL host-name test data down to 0.56. We explain why rather than hide it.
- 72 unit tests, reproducible `make data / train / eval`, split manifests with hashes, and model cards.
- Privacy: nothing stored, no logging, links never opened.

### 8. Close (M, 15 s)
"FraudShield gives an explained second opinion, but it can't guarantee safety. The advice always ends the same way: check through the organisation's official app, website or phone number."

---

**Likely questions**
- *Why not just use an LLM?* Labelled data, measurable metrics, offline and private, about 1 s per check, and explanations tied to the model's actual inputs.
- *Why is DistilBERT only tied with the baseline?* Only about 70 spam and 70 scam validation messages, and noisy spam/scam labels. DistilBERT has higher scam recall, which is our priority.
- *Why bands and not percentages?* The scores aren't calibrated, and DistilBERT's sit near 0 or 1, so a percentage would overstate certainty.
- *Roman Urdu?* Not supported. It's the clearest next step, along with a WhatsApp bot.
