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
from fraudshield.inference.context_rules import Context, context_floor
from fraudshield.inference.conversation import (
    Turn,
    build_pieces,
    guess_me,
    parse_conversation,
    score_texts,
    token_budget,
    word_token_counter,
)
from fraudshield.inference.conversation_rules import conversation_floor
from fraudshield.inference.extract_urls import extract_urls
from fraudshield.inference.risk import BANDS, band, max_band
from fraudshield.utils import load_config, resolve

MAX_CHARS = 2000
MAX_CONVERSATION_CHARS = 8000
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
    parts_note: str = ""  # set when a long message was read in overlapping parts


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
    model_overall: str = "Low"  # rating from the models (+ link rules) before user context
    context_notes: list = field(default_factory=list)  # why the user's context raised it

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TurnResult:
    index: int
    speaker: str | None
    text: str
    is_me: bool
    band: str | None = None       # None for your own messages (not scored)
    predicted: str | None = None
    score: float | None = None


@dataclass
class ConversationReport:
    overall: str
    headline: str
    model_overall: str
    turns: list[TurnResult]
    top: dict | None              # highest-scoring window: start, end, text, predicted, score, part
    urls: list[UrlResult] = field(default_factory=list)
    rule_notes: list = field(default_factory=list)     # cross-message patterns
    context_notes: list = field(default_factory=list)  # from the optional context questions
    signals: list = field(default_factory=list)        # scam signals in the top window
    capped: bool = False
    advice: str = ""
    me: str | None = None
    format: str = ""
    truncated: bool = False
    timings_ms: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class FraudShield:
    def __init__(self, message_model, url_model, thresholds: dict, spam_max_band: str = "Medium",
                 conversation_thresholds: dict | None = None):
        self.message_model = message_model
        self.conversation_thresholds = conversation_thresholds  # None: conversation mode unavailable
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
        conv = None
        for p in (mdir / "conversation_thresholds.json", resolve("reports/metrics/conversation_thresholds.json")):
            if p.exists():
                conv = json.loads(p.read_text())
                break
        if conv is not None and conv.get("model") != expected:
            conv = None  # tuned for a different message model
        return cls(msg_model, url_model, thresholds, mcfg.get("spam_max_band", "Medium"), conv)

    # ------------------------------------------------------------------ analysis
    def analyse(self, text: str, context: Context | None = None) -> Report:
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

        model_overall = max_band(message.band if message else None, *[u.band for u in url_results])
        # Optional user context: can only raise the rating, never lower it (context_rules.py).
        floor, context_notes = context_floor(text, context, has_urls=bool(found))
        overall = max_band(model_overall, floor)
        headline = _headline(model_overall, message, url_results)
        if overall != model_overall:
            headline = _context_headline(overall)
        timings["total"] = _ms(t0)
        return Report(
            overall=overall,
            headline=headline,
            message=message,
            urls=url_results,
            truncated=truncated,
            advice=ADVICE[overall],
            timings_ms=timings,
            models={"message": getattr(self.message_model, "name", "?"),
                    "url": "Random Forest (lexical + structural features)"},
            model_overall=model_overall,
            context_notes=context_notes,
        )

    def analyse_conversation(self, text: str, me: str | None = "auto",
                             context: Context | None = None) -> ConversationReport:
        """Score a pasted chat: windows of the other person's messages (conversation.py),
        cross-message rules, links, optional context. `me` = your name in the chat, so your
        own messages aren't scored; "auto" guesses ("You"/"Me"), None scores everyone."""
        if self.conversation_thresholds is None:
            raise RuntimeError("Conversation thresholds are missing. Run `make conversation-thresholds`.")
        t0 = time.perf_counter()
        truncated = len(text or "") > MAX_CONVERSATION_CHARS
        turns, fmt = parse_conversation((text or "")[:MAX_CONVERSATION_CHARS])
        turns = [Turn(t.index, t.speaker, normalise_text(t.text)) for t in turns]
        turns = [t for t in turns if t.text]
        if me == "auto":
            me = guess_me(turns)
        other = [t for t in turns if not (me and t.speaker == me)]
        if not other:
            return ConversationReport(overall="Low", headline="No messages from the other person to analyse.",
                                      model_overall="Low", turns=[], top=None, advice=ADVICE["Low"], me=me,
                                      format=fmt, truncated=truncated)

        conv = self.conversation_thresholds
        pieces = build_pieces(other, word_token_counter(self.message_model),
                              token_budget(self.message_model), max_messages=conv["max_messages"])
        scores, labels = score_texts(self.message_model, [p.text for p in pieces])
        bands = [self._message_band(s, lab, conv) for s, lab in zip(scores, labels)]
        # Top window = highest band after caps (so a capped spam window never hides a scam
        # window that reaches High), then highest score, then fewest messages (most precise).
        i = max(range(len(pieces)), key=lambda j: (BANDS.index(bands[j][0]), scores[j], -pieces[j].n_messages))
        top_piece, top_score, top_label = pieces[i], scores[i], labels[i]
        model_band, capped = bands[i]

        per_turn: dict[int, tuple[float, str]] = {}
        for p, s, lab in zip(pieces, scores, labels):
            if p.n_messages == 1 and (p.start not in per_turn or s > per_turn[p.start][0]):
                per_turn[p.start] = (s, lab)
        turn_results = []
        for t in turns:
            if me and t.speaker == me:
                turn_results.append(TurnResult(t.index, t.speaker, t.text, True))
            else:
                s, lab = per_turn[t.index]
                tb = self._message_band(s, lab, conv)[0]
                turn_results.append(TurnResult(t.index, t.speaker, t.text, False, tb, lab, s))

        found, seen = [], set()
        for t in other:
            for f in extract_urls(t.text, limit=MAX_URLS):
                if f.url not in seen and len(found) < MAX_URLS:
                    seen.add(f.url)
                    found.append(f)
        url_results = [self._analyse_url(f) for f in found]

        model_overall = max_band(model_band, *[u.band for u in url_results])
        rule_floor, rule_notes = conversation_floor(other)
        ctx_floor, context_notes = context_floor(" ".join(t.text for t in other), context, has_urls=bool(found))
        overall = max_band(model_overall, rule_floor, ctx_floor)

        span = (f"Message {top_piece.start}" if top_piece.start == top_piece.end
                else f"Messages {top_piece.start}–{top_piece.end}")
        if overall == model_overall:
            if overall == "Low":
                headline = "No strong warning signs found in this conversation."
            elif model_band == overall and top_label != "ham":
                kind = {"scam": "look like a scam" if overall == "High" else "may be a scam",
                        "spam": "look like spam or a promotion"}[top_label]
                headline = f"{span} {kind}."
            else:
                headline = _headline(overall, None, url_results)
        elif rule_floor == overall:
            headline = "A scam pattern runs across this conversation."
        else:
            headline = _context_headline(overall)

        return ConversationReport(
            overall=overall, headline=headline, model_overall=model_overall, turns=turn_results,
            top={"start": top_piece.start, "end": top_piece.end, "text": top_piece.text,
                 "predicted": top_label, "score": top_score, "part": top_piece.part,
                 "n_messages": top_piece.n_messages},
            urls=url_results, rule_notes=rule_notes, context_notes=context_notes,
            signals=find_signals(top_piece.text), capped=capped, advice=ADVICE[overall], me=me,
            format=fmt, truncated=truncated, timings_ms={"total": _ms(t0)},
            models={"message": getattr(self.message_model, "name", "?"),
                    "windows": f"up to {conv['max_messages']} messages per window"},
        )

    def _analyse_message(self, text: str) -> MessageResult:
        occ = occlusion(text, self.message_model.predict_log_proba, self.message_model.labels)
        b, capped = self._message_band(occ["base_score"], occ["predicted"])
        predicted, score, parts_note = occ["predicted"], occ["base_score"], ""
        # Longer than the model reads at once (128 tokens): also score overlapping parts and
        # keep the higher band (raise-only). No test message is that long, so the evaluated
        # results are unchanged.
        if hasattr(self.message_model, "max_length"):
            pieces = build_pieces([Turn(1, None, text)], word_token_counter(self.message_model),
                                  token_budget(self.message_model))
            if len(pieces) > 1:
                scores, labels = score_texts(self.message_model, [p.text for p in pieces])
                i = max(range(len(scores)), key=scores.__getitem__)
                parts_note = (f"This message is longer than the model reads at once, so it was also read "
                              f"in {len(pieces)} overlapping parts.")
                pb, pcapped = self._message_band(scores[i], labels[i])
                if BANDS.index(pb) > BANDS.index(b):
                    b, capped, predicted, score = pb, pcapped, labels[i], scores[i]
                    parts_note += f" Part {i + 1} scored highest and sets the rating."
        return MessageResult(
            band=b, predicted=predicted, score=score, probs=occ["probs"],
            top_words=[{k: w[k] for k in ("word", "start", "end", "importance")} for w in occ["top_words"]],
            distributed=occ["distributed"], signals=find_signals(text), capped=capped,
            parts_note=parts_note,
        )

    def _message_band(self, score: float, predicted: str, thresholds: dict | None = None) -> tuple[str, bool]:
        """Band for one score, with the app caps: spam at most spam_max_band, and an argmax of
        ham is never High. Returns (band, capped). Default thresholds: single message."""
        b = band(score, thresholds or self.thresholds["message"])
        capped = False
        if predicted == "spam" and BANDS.index(b) > BANDS.index(self.spam_max_band):
            b, capped = self.spam_max_band, True
        if predicted == "ham" and b == "High":
            b = "Medium"
        return b, capped

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


def _context_headline(overall: str) -> str:
    if overall == "High":
        return "The context you gave makes this look like a scam."
    return "The context you gave makes this worth checking."


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
