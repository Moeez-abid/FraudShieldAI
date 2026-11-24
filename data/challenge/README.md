# Challenge set: labelling protocol

This is an independent evaluation set. It is **never used for training or tuning**, only for final reporting.

## Targets
- `messages.csv`: 60–100 recent English messages (about 50 % scam, 25 % spam, 25 % ham).
- `urls.csv`: 60–100 URLs, about half phishing and half benign.

Neither may come from the UCI, Mendeley or PhiUSIIL datasets. Run `dedup_key` against the training data before freezing.

## Sources
- Messages: your own or volunteers' phones (shared with permission), and public scam-awareness pages (bank, telecom and government warnings). Remove names, phone numbers and account numbers: replace them with `<NAME>`, `<PHONE>`, `<ACCOUNT>`.
- Phishing URLs: recent entries from public feeds such as OpenPhish or PhishTank. **Copy the text only and never open the links.** Defang them when storing (`hxxp://`, `[.]`). The loader will refang them.
- Benign URLs: real pages from popular and long-tail sites, including deep links with paths and queries, not just homepages. PhiUSIIL's benign URLs are mostly homepages, so this tests whether the model learned that artefact.

## Columns
`messages.csv`: `id, text, source_note, label_moeez, label_ijtaba, label_final, notes`
`urls.csv`: `id, url_defanged, source_note, label_moeez, label_ijtaba, label_final, notes`

## Process
1. Each member labels every item **independently** without seeing the other's labels (ham / spam / scam; benign / phishing).
2. Compute Cohen's κ on the independent labels and report it.
3. Discuss disagreements and record `label_final` and a short note.
4. Freeze the files with a commit. Don't edit them after Week 3.

**Label definitions.** *Spam:* unsolicited promotion with no deception about who is sending it. *Scam:* tries to deceive the reader into giving money, credentials or personal data, or into following a malicious link, often by impersonating someone.
