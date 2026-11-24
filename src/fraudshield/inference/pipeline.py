"""End-to-end analysis: one call from pasted text to a report.

    shield = FraudShield.load()
    report = shield.analyse("URGENT: verify your account at hxxp://bad[.]top/login")

Privacy: nothing is logged or written to disk, and URLs are never fetched.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib

from fraudshield.data.clean_messages import normalise_text
from fraudshield.explain.text_occlusion import occlusion
from fraudshield.explain.text_signals import find_signals
from fraudshield.explain.url_shap import UrlExplainer
from fraudshield.features.url_features import registered_domain
from fraudshield.inference.extract_urls import extract_urls
from fraudshield.inference.risk import BANDS, band, max_band
from fraudshield.utils import load_config, resolve

MAX_CHARS = 2000
MAX_URLS = 5
URL_ONLY_MAX_OTHER_WORDS = 2  # "check this: <url>" is treated as a URL-only input

# Link shorteners hide the real destination. Rule (outside the model): rate at least
# Medium. Broader than the model's SHORTENERS feature list, which stays unchanged so
# the trained model's inputs don't change. See reports/error_analysis.md, 2B.
SHORTENER_DOMAINS = frozenset({
    "bit.ly", "bitly.com", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "s.id", "t.ly", "v.gd",
    "short.gy", "tinu.be", "href.li", "shorturl.asia", "u.to", "clck.ru", "qr.ae", "lnkd.in",
    "rebrandly.com", "bl.ink", "soo.gd", "shorte.st", "adf.ly", "x.co", "trib.al", "linktr.ee",
})

ADVICE = {
    "High": ("Don't reply, call back, click links or share any codes. If it claims to be from "
             "an organisation you use, contact it through its official app, website or phone number."),
    "Medium": ("Be careful. Don't share passwords, PINs or one-time codes. Before acting, check "
               "the request through the organisation's official app, website or phone number."),
    "Low": ("No strong warning signs, but a Low rating doesn't guarantee safety. Never share "
            "passwords, PINs or one-time codes by message."),
}


@dataclass
class MessageResult:
    band: str
    predicted: str
    score: float
    probs: dict
    top_words: list
    distributed: bool
    signals: list
    capped: bool = False


@dataclass
class UrlResult:
    original: str
    url: str
    band: str
    score: float
    reasons: list
    defanged: bool
    scheme_given: bool
    rule_notes: list = field(default_factory=list)  # rule-based adjustments, shown separately


@dataclass
class Report:
    overall: str
    headline: str
    message: MessageResult | None
    urls: list[UrlResult] = field(default_factory=list)
    truncated: bool = False
    advice: str = ""
    timings_ms: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class FraudShield:
    def __init__(self, message_model, url_model, thresholds: dict, spam_max_band: str = "Medium"):
        self.message_model = message_model
        self.url_model = url_model
        self.url_explainer = UrlExplainer(url_model)
        self.thresholds = thresholds
        self.spam_max_band = spam_max_band

    @classmethod
    def load(cls, prefer_distilbert: bool = True) -> "FraudShield":
        from fraudshield.inference.message_model import (
            BaselineMessageModel,
            DistilBertMessageModel,
        )

        mcfg, ucfg = load_config("message"), load_config("url")
        mdir = resolve(mcfg["paths"]["model_dir"])
        msg_model = None
        if prefer_distilbert and (mdir / "distilbert" / "fraudshield.json").exists():
            try:
                msg_model = DistilBertMessageModel(mdir / "distilbert")
            except ImportError:  # torch/transformers not installed: fall back
                msg_model = None
        if msg_model is None:
            msg_model = BaselineMessageModel(mdir / "baseline_tfidf_lr.joblib")

        view = ucfg.get("serving_view", "mixed")
        url_model = joblib.load(mdir / f"url_rf_{view}.joblib")
        thresholds = json.loads((mdir / "risk_thresholds.json").read_text())
        expected = "distilbert" if isinstance(msg_model, DistilBertMessageModel) else "baseline"
        if thresholds["message"]["model"] != expected:
            raise RuntimeError(
                f"risk_thresholds.json was computed for '{thresholds['message']['model']}' but the "
                f"loaded message model is '{expected}'. Run `make thresholds`."
            )
        return cls(msg_model, url_model, thresholds, mcfg.get("spam_max_band", "Medium"))

    # ------------------------------------------------------------------ analysis
    def analyse(self, text: str) -> Report:
        t0 = time.perf_counter()
        timings = {}
        text = normalise_text(text or "")
        truncated = len(text) > MAX_CHARS
        text = text[:MAX_CHARS]
        if not text:
            return Report(overall="Low", headline="Nothing to analyse.", message=None,
                          advice=ADVICE["Low"])

        found = extract_urls(text, limit=MAX_URLS)
        timings["extract"] = _ms(t0)

        remaining = text
        for f in sorted(found, key=lambda f: f.start, reverse=True):
            remaining = remaining[:f.start] + remaining[f.end:]
        url_only = bool(found) and len(remaining.split()) <= URL_ONLY_MAX_OTHER_WORDS

        message = None
        if not url_only:
            t = time.perf_counter()
            message = self._analyse_message(text)
            timings["message"] = _ms(t)

        t = time.perf_counter()
        url_results = [self._analyse_url(f) for f in found]
        timings["urls"] = _ms(t)

        overall = max_band(message.band if message else None, *[u.band for u in url_results])
        timings["total"] = _ms(t0)
        return Report(
            overall=overall,
            headline=_headline(overall, message, url_results),
            message=message,
            urls=url_results,
            truncated=truncated,
            advice=ADVICE[overall],
            timings_ms=timings,
            models={"message": getattr(self.message_model, "name", "?"),
                    "url": "Random Forest (lexical + structural features)"},
        )

    def _analyse_message(self, text: str) -> MessageResult:
        occ = occlusion(text, self.message_model.predict_log_proba, self.message_model.labels)
        b = band(occ["base_score"], self.thresholds["message"])
        capped = False
        if occ["predicted"] == "spam" and BANDS.index(b) > BANDS.index(self.spam_max_band):
            b, capped = self.spam_max_band, True
        if occ["predicted"] == "ham" and b == "High":  # argmax says benign: never "High"
            b = "Medium"
        return MessageResult(
            band=b, predicted=occ["predicted"], score=occ["base_score"], probs=occ["probs"],
            top_words=[{k: w[k] for k in ("word", "start", "end", "importance")} for w in occ["top_words"]],
            distributed=occ["distributed"], signals=find_signals(text), capped=capped,
        )

    def _analyse_url(self, f) -> UrlResult:
        skip = frozenset() if f.scheme_given else frozenset({"is_https", "has_scheme"})
        exp = self.url_explainer.explain(f.url, skip=skip)
        b = band(exp["score"], self.thresholds["url"])
        notes = []
        if registered_domain(f.url) in SHORTENER_DOMAINS:
            notes.append("Uses a link shortener, which hides the real destination. "
                         "Rated at least Medium by rule.")
            b = max_band(b, "Medium")
        return UrlResult(
            original=f.original, url=f.url, band=b, score=exp["score"], reasons=exp["reasons"],
            defanged=f.defanged, scheme_given=f.scheme_given, rule_notes=notes,
        )


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


def _headline(overall: str, message: MessageResult | None, urls: list[UrlResult]) -> str:
    if overall == "Low":
        return "No strong warning signs found."
    sure = overall == "High"
    parts = []
    if message and message.band == overall:
        kind = {
            "scam": "looks like a scam" if sure else "may be a scam",
            "spam": "looks like spam or a promotion",
            "ham": "has some unusual features",
        }[message.predicted]
        parts.append(f"The message {kind}")
    risky = [u for u in urls if u.band == overall]
    if risky:
        n = len(risky)
        verb = ("looks" if n == 1 else "look") + (" suspicious" if sure else " worth checking")
        parts.append(f"{n} link{'s' if n > 1 else ''} {verb}")
    return " and ".join(parts) + "."
