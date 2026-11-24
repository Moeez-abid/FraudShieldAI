"""FraudShield AI: Streamlit app.

Run:  make app   (or: streamlit run app/streamlit_app.py)

Privacy: inputs are analysed in memory only. Nothing is logged or saved, and links
are never opened. URLs are always shown as plain text, never as clickable links.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from fraudshield.inference.pipeline import FraudShield, Report
from fraudshield.utils import resolve

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️", layout="wide")

BAND_STYLE = {
    "Low": ("#1a7f37", "rgba(26,127,55,0.12)", "Low risk"),
    "Medium": ("#9a6700", "rgba(212,167,44,0.18)", "Medium risk"),
    "High": ("#cf222e", "rgba(207,34,46,0.13)", "High risk"),
}
LABEL_TEXT = {"ham": "Normal message", "spam": "Spam / promotion", "scam": "Scam / phishing"}

SAMPLES = {
    "Bank scam": ("URGENT: Your HBL account has been suspended due to unusual activity. "
                  "Verify your account within 24 hours at hxxp://hbl-secure-verify[.]top/login "
                  "or it will be permanently closed."),
    "Prize scam": ("Congratulations! You have won Rs 50,000 in the Jazz lucky draw. Send your CNIC "
                   "and pay Rs 500 processing fee to claim your prize."),
    "Delivery notice": "Your Daraz order #48213 has been shipped and will arrive on Thursday.",
    "Promotion": "50% off all shoes this weekend only at Outfitters. Shop now at outfitters.com.pk",
    "Normal chat": "Hey, are we still meeting for lunch tomorrow at 1? I'll book a table.",
    "Just a link": "http://paypal.com-secure-login.xyz/verify?session=88123",
}

st.markdown("""
<style>
  .block-container {max-width: 1150px; padding-top: 2rem;}
  div[data-testid="stTextArea"] textarea {font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models (first run only)...")
def get_shield() -> FraudShield:
    return FraudShield.load()


def badge(band: str, size: str = "0.8rem") -> str:
    fg, bg, label = BAND_STYLE[band]
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
            f'background:{bg};color:{fg};font-weight:600;font-size:{size};'
            f'border:1px solid {fg}33">{label}</span>')


def verdict_banner(report: Report) -> None:
    fg, bg, label = BAND_STYLE[report.overall]
    icon = {"Low": "✓", "Medium": "!", "High": "⚠"}[report.overall]
    st.html(f"""
    <div style="border-left:6px solid {fg};background:{bg};border-radius:10px;
                padding:16px 20px;margin:8px 0 4px 0;font-family:inherit">
      <div style="font-size:1.35rem;font-weight:700;color:{fg}">{icon} {label}</div>
      <div style="margin-top:4px;font-size:1rem;color:inherit">{html.escape(report.headline)}</div>
    </div>""")


def highlighted_text(text: str, words: list[dict]) -> str:
    """Escape the message and highlight the top words by importance."""
    if not words:
        return html.escape(text)
    top = max(w["importance"] for w in words)
    spans = sorted(words, key=lambda w: w["start"])
    out, pos = [], 0
    for w in spans:
        alpha = 0.25 + 0.55 * (w["importance"] / top)
        out.append(html.escape(text[pos:w["start"]]))
        out.append(f'<mark style="background:rgba(207,34,46,{alpha:.2f});color:inherit;'
                   f'padding:1px 3px;border-radius:4px">{html.escape(text[w["start"]:w["end"]])}</mark>')
        pos = w["end"]
    out.append(html.escape(text[pos:]))
    return "".join(out)


def message_panel(report: Report, text: str) -> None:
    m = report.message
    st.subheader("Message")
    if m is None:
        st.caption("Only a link was entered, so the message model was skipped.")
        return
    st.html(f'{badge(m.band)} &nbsp; <span style="opacity:.8">Model reading: '
            f'<b>{LABEL_TEXT[m.predicted]}</b></span>')
    if m.capped:
        st.caption("Rated as promotion, so it's capped at Medium: annoying, but not deceptive.")

    st.html(f'<div style="border:1px solid rgba(128,128,128,.3);border-radius:8px;padding:12px 14px;'
            f'line-height:1.7;white-space:pre-wrap;word-break:break-word">'
            f'{highlighted_text(text, m.top_words)}</div>')
    if m.top_words:
        st.caption("Highlighted words pushed the model toward this rating most "
                   "(measured by removing each word and re-scoring).")
    elif m.distributed:
        st.info("No single word decides this. The message as a whole matches scam patterns, "
                "so removing any one word doesn't change the verdict.")

    if m.signals:
        st.markdown("**Common scam signals found** · *rule-based checks, separate from the model*")
        for s in m.signals:
            phrases = ", ".join(f"“{html.escape(p)}”" for p in s["matches"][:4])
            st.html(f'<div style="margin:6px 0;padding:8px 12px;border-radius:8px;'
                    f'background:rgba(128,128,128,.08)"><b>{html.escape(s["title"])}</b>'
                    f' <span style="opacity:.75">({phrases})</span><br>'
                    f'<span style="opacity:.85;font-size:.9rem">{html.escape(s["explanation"])}</span></div>')
    elif m.predicted != "ham":
        st.caption("No rule-based scam signals matched.")


def url_panel(report: Report) -> None:
    st.subheader("Links")
    if not report.urls:
        st.caption("No links found in the text.")
        return
    for u in report.urls:
        notes = []
        if u.defanged:
            notes.append("written in a disguised form in the message (e.g. hxxp, [.]); restored for analysis")
        if not u.scheme_given:
            notes.append("no http(s):// was given, so http:// was assumed")
        rules = "".join(f'<div style="font-size:.85rem;margin-top:6px;padding:6px 8px;border-radius:6px;'
                        f'background:rgba(212,167,44,.15)"><b>Rule:</b> {html.escape(n)}</div>'
                        for n in u.rule_notes)
        reasons = "".join(
            f'<li><span style="color:{"#cf222e" if r["contribution"] > 0 else "#1a7f37"};font-weight:700">'
            f'{"▲" if r["contribution"] > 0 else "▼"}</span> {html.escape(r["text"])}</li>'
            for r in u.reasons
        ) or "<li>No single feature stood out.</li>"
        st.html(f"""
        <div style="border:1px solid rgba(128,128,128,.3);border-radius:10px;padding:12px 14px;margin-bottom:10px">
          <div>{badge(u.band)}</div>
          <div style="margin:8px 0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88rem;
                      word-break:break-all;background:rgba(128,128,128,.1);padding:6px 8px;border-radius:6px">
            {html.escape(u.url)}</div>
          {"".join(f'<div style="font-size:.82rem;opacity:.75">ⓘ {html.escape(n)}</div>' for n in notes)}
          {rules}
          <div style="font-size:.9rem;margin-top:6px"><b>Why</b>
            <span style="opacity:.7">(▲ raises risk, ▼ lowers risk)</span>
            <ul style="margin:4px 0 0 0;padding-left:18px">{reasons}</ul></div>
        </div>""")
    st.caption("Links are analysed from their text only. FraudShield never opens them.")


def sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🛡️ FraudShield AI")
        st.markdown("Second opinion on suspicious **English** messages and links. "
                    "Student prototype (FAST-NUCES GenAI project, Fall 2026).")
        st.markdown("**Privacy**: nothing you paste is stored or logged, and links are never opened.")
        with st.expander("How it works"):
            st.markdown(
                "- **Message**: a fine-tuned DistilBERT model reads the text (ham / spam / scam). "
                "Highlighted words come from re-scoring the message with each word removed.\n"
                "- **Scam signals**: simple pattern rules (urgency, OTP requests...). Shown separately.\n"
                "- **Links**: a Random Forest looks at the link's structure (domain, subdomains, "
                "characters, keywords). Reasons come from SHAP.\n"
                "- **Risk bands** are set on validation data. Scores are not probabilities, so "
                "no percentages are shown.\n"
                "- **Overall** = the highest risk among the message and its links.")
        with st.expander("Limitations"):
            st.markdown(
                "- Trained on public SMS and URL datasets (2011–2024); new scam styles may be missed.\n"
                "- English only. Roman Urdu and mixed-language messages are not supported.\n"
                "- Normal delivery notices and bank alerts can be flagged.\n"
                "- Phishing on ordinary-looking or hacked domains is often missed: the link check only "
                "sees the text of the link.\n"
                "- Legitimate links on less common sites can be rated Medium.\n"
                "- **A Low rating is not a guarantee of safety.**")
        thr = resolve("models/risk_thresholds.json")
        if thr.exists():
            with st.expander("Validation metrics"):
                t = json.loads(thr.read_text())
                st.markdown(
                    f"- Message ({t['message']['model']}): at High, precision "
                    f"{t['message']['at_high']['precision']:.0%}, recall {t['message']['at_high']['recall']:.0%}\n"
                    f"- Links ({t['url']['model']}, {t['url'].get('slice', 'all')}): at High, precision "
                    f"{t['url']['at_high']['precision']:.0%}, recall {t['url']['at_high']['recall']:.0%}")


def main() -> None:
    sidebar()
    st.title("Is this message a scam?")
    st.caption("Paste an SMS, WhatsApp message, email excerpt or link. "
               "You get a risk rating with the reasons behind it.")

    if "text" not in st.session_state:
        st.session_state.text = ""
    st.markdown("**Try an example:**")
    items = list(SAMPLES.items())
    for row in (items[:3], items[3:]):
        for col, (name, sample) in zip(st.columns(3), row):
            if col.button(name, use_container_width=True):
                st.session_state.text = sample
                st.session_state.run = True

    text = st.text_area("Message or link", key="text", height=130, max_chars=2000,
                        placeholder="e.g. Your account has been suspended. Verify at ...",
                        label_visibility="collapsed")
    run = st.button("Analyse", type="primary") or st.session_state.pop("run", False)

    if not run:
        return
    if not text.strip():
        st.warning("Paste a message or link first.")
        return

    try:
        shield = get_shield()
    except FileNotFoundError as exc:
        st.error(f"Model files are missing ({Path(str(exc.filename or exc)).name}). "
                 "Run `make train`, or copy `models/` from the shared Drive folder (see README).")
        return
    except RuntimeError as exc:  # thresholds computed for a different message model
        st.error(str(exc))
        return

    with st.spinner("Analysing..."):
        report = shield.analyse(text)

    verdict_banner(report)
    if report.truncated:
        st.caption("Only the first 2,000 characters were analysed.")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        message_panel(report, report_text(text))
    with right:
        url_panel(report)

    st.html(f'<div style="margin-top:14px;padding:12px 16px;border-radius:10px;'
            f'background:rgba(9,105,218,.09);border-left:5px solid #0969da">'
            f'<b>What to do</b><br>{html.escape(report.advice)}</div>')
    st.caption(f"Analysed in {report.timings_ms.get('total', 0):.0f} ms · "
               f"message model: {report.models.get('message')} · nothing was stored.")


def report_text(text: str) -> str:
    """The exact text the pipeline analysed (normalised, truncated), so highlight
    offsets line up with what the model saw."""
    from fraudshield.data.clean_messages import normalise_text
    from fraudshield.inference.pipeline import MAX_CHARS
    return normalise_text(text)[:MAX_CHARS]


main()
