"""End-to-end pipeline tests with small stand-in models (no trained artefacts needed)."""

import numpy as np
import pytest

from fraudshield.features.url_features import extract_frame
from fraudshield.inference.pipeline import MAX_CHARS, FraudShield
from fraudshield.models.url_rf import make_pipeline

LABELS = ["ham", "spam", "scam"]


class FakeMessageModel:
    """scam if 'verify' in text, spam if 'sale' in text, else ham (as log-probs)."""
    name = "fake"
    labels = LABELS

    def predict_log_proba(self, texts):
        rows = []
        for t in texts:
            t = t.lower()
            if "verify" in t:
                p = [0.01, 0.01, 0.98]
            elif "sale" in t:
                p = [0.005, 0.99, 0.005]
            else:
                p = [0.98, 0.01, 0.01]
            rows.append(p)
        return np.log(np.array(rows))

    def predict_proba(self, texts):
        return np.exp(self.predict_log_proba(texts))


@pytest.fixture(scope="module")
def shield():
    benign = ["https://www.example.com/about", "https://news.site.org/world/today",
              "https://shop.store.com/item/123", "https://www.wiki.org/page"] * 5
    phish = ["http://10.0.0.5/login.php", "http://paypal.secure-verify.xyz/account",
             "http://apple.com-id-check.top/signin", "http://192.168.1.9/bank/verify"] * 5
    rf = make_pipeline({"seed": 0, "random_forest": {"n_estimators": 20}}, None, 1)
    rf.fit(extract_frame(benign + phish), np.array([0] * 20 + [1] * 20))
    thresholds = {"message": {"model": "fake", "medium": 0.3, "high": 0.9},
                  "url": {"model": "rf", "medium": 0.4, "high": 0.7}}
    return FraudShield(FakeMessageModel(), rf, thresholds, spam_max_band="Medium")


def test_scam_message_with_link(shield):
    r = shield.analyse("Please verify your account at hxxp://paypal.secure-verify[.]xyz/login now")
    assert r.overall == "High"
    assert r.message.predicted == "scam" and r.message.band == "High"
    assert len(r.urls) == 1 and r.urls[0].defanged
    assert r.urls[0].url == "http://paypal.secure-verify.xyz/login"
    assert r.headline == "The message looks like a scam and 1 link looks suspicious."


def test_benign_message(shield):
    r = shield.analyse("See you at lunch tomorrow")
    assert r.overall == "Low" and r.message.band == "Low" and r.urls == []
    assert r.message.top_words == []


def test_spam_is_capped_at_medium(shield):
    r = shield.analyse("Big sale this weekend, everything half price")
    assert r.message.predicted == "spam"
    assert r.message.band == "Medium" and r.message.capped
    assert r.headline == "The message looks like spam or a promotion."
    assert r.advice.startswith("Be careful")


def test_low_advice_differs_from_high(shield):
    low = shield.analyse("See you at lunch tomorrow")
    high = shield.analyse("verify at http://10.0.0.5/login.php")
    assert low.headline == "No strong warning signs found."
    assert low.advice != high.advice and "doesn't guarantee" in low.advice


def test_url_only_input_skips_message_model(shield):
    r = shield.analyse("check this http://10.0.0.7/login.php")
    assert r.message is None
    assert r.urls[0].band == "High"
    assert "message" not in r.timings_ms


def test_bare_url_does_not_blame_missing_https(shield):
    r = shield.analyse("10.0.0.9/login.php")
    assert not r.urls[0].scheme_given
    assert all(x["feature"] not in {"is_https", "has_scheme"} for x in r.urls[0].reasons)


def test_empty_and_long_input(shield):
    assert shield.analyse("   ").overall == "Low"
    r = shield.analyse("hello " * 1000)
    assert r.truncated and r.message is not None
    assert len("hello " * 1000) > MAX_CHARS


def test_report_is_json_serialisable(shield):
    import json
    json.dumps(shield.analyse("verify at http://10.0.0.5/login.php").to_dict())


def test_shortener_rule_floors_at_medium(shield):
    r = shield.analyse("https://www.example.com/about and https://tinyurl.com/abc123")
    by_url = {u.url: u for u in r.urls}
    short = by_url["https://tinyurl.com/abc123"]
    assert short.band in ("Medium", "High") and short.rule_notes
    assert by_url["https://www.example.com/about"].rule_notes == []


def test_context_only_raises_and_no_context_is_identical(shield):
    from fraudshield.inference.context_rules import NO, SENDER_CONTACT, SENDER_UNKNOWN, YES, Context

    text = "Hi Ammi it's me, this is my new number"
    base = shield.analyse(text)
    assert shield.analyse(text, context=Context()).to_dict() | {"timings_ms": {}} == \
        base.to_dict() | {"timings_ms": {}}
    assert base.overall == "Low" and base.context_notes == []

    raised = shield.analyse(text, context=Context(sender=SENDER_UNKNOWN))
    assert raised.overall == "Medium" and raised.model_overall == "Low"
    assert raised.context_notes and raised.headline.startswith("The context you gave")
    assert raised.message.band == "Low"  # the model's own reading is untouched

    high = shield.analyse("verify at http://10.0.0.5/login.php",
                          context=Context(sender=SENDER_CONTACT, expected=YES, has_account=YES))
    assert high.overall == "High"  # reassuring answers never lower

    pre = shield.analyse("Please verify your account", context=Context(expected=NO))
    assert pre.overall == "High"


@pytest.fixture(scope="module")
def conv_shield(shield):
    from fraudshield.inference.pipeline import FraudShield
    conv = {"model": "fake", "medium": 0.3, "high": 0.9, "max_messages": 3}
    return FraudShield(shield.message_model, shield.url_model, shield.thresholds,
                       spam_max_band="Medium", conversation_thresholds=conv)


def test_conversation_scores_other_person_only(conv_shield):
    chat = "Them: hello there\nYou: please verify your account\nThem: see you soon"
    r = conv_shield.analyse_conversation(chat, me="You")
    assert r.overall == "Low" and r.me == "You"
    assert [t.is_me for t in r.turns] == [False, True, False] and r.turns[1].band is None
    r = conv_shield.analyse_conversation(chat, me=None)  # no "me" chosen -> everything scored
    assert r.overall == "High" and r.top["start"] == 2


def test_conversation_rules_raise_and_links(conv_shield):
    chat = ("Mum: Hi beta, it's me, this is my new number\nYou: ok\n"
            "Mum: Please send 20,000 on JazzCash today\nMum: details at hxxp://10.0.0.5/login.php")
    r = conv_shield.analyse_conversation(chat, me="You")
    assert r.overall == "High" and r.rule_notes
    assert r.urls and r.urls[0].url == "http://10.0.0.5/login.php"


def test_conversation_needs_thresholds(shield):
    with pytest.raises(RuntimeError):
        shield.analyse_conversation("a: hi\nb: hello")


def test_long_message_is_read_in_parts_raise_only(shield):
    class TokModel(type(shield.message_model)):
        max_length = 12  # tiny budget so a normal sentence needs several parts

    from fraudshield.inference.pipeline import FraudShield
    s = FraudShield(TokModel(), shield.url_model, shield.thresholds)
    text = "hello " * 30 + "please verify now"
    r = s.analyse(text)
    assert r.message.parts_note and r.message.band == "High"
    short = s.analyse("hello there friend")
    assert short.message.parts_note == ""
