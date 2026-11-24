import numpy as np
import pandas as pd
import pytest

from fraudshield.explain.text_occlusion import occlusion
from fraudshield.explain.text_signals import find_signals

LABELS = ["ham", "spam", "scam"]


def fake_proba(texts):
    """Log-probs: scam probability 0.9 if the word 'verify' is present, else 0.1."""
    out = []
    for t in texts:
        p_scam = 0.9 if "verify" in t.lower() else 0.1
        out.append([1 - p_scam, 1e-12, p_scam])
    return np.log(np.array(out))


def test_occlusion_ranks_the_driving_word_first():
    r = occlusion("Please verify your account now", fake_proba, LABELS)
    assert r["predicted"] == "scam"
    assert r["top_words"][0]["word"] == "verify"
    assert r["top_words"][0]["importance"] == pytest.approx(2 * np.log(9))  # log-odds 0.9 vs 0.1
    # spans point back into the original text
    w = r["top_words"][0]
    assert "Please verify your account now"[w["start"]:w["end"]] == "verify"


def test_occlusion_benign_has_no_top_words():
    r = occlusion("see you at lunch", fake_proba, LABELS)
    assert r["predicted"] == "ham" and r["top_words"] == [] and not r["distributed"]


def test_occlusion_flags_distributed_evidence():
    def redundant(texts):  # scam if ANY two of the cue words remain
        cues = ("won", "prize", "fee")
        return np.log(np.array([[0.05, 1e-9, 0.95] if sum(c in t for c in cues) >= 2
                                else [0.9, 1e-9, 0.1] for t in texts]))
    r = occlusion("you won a prize pay the fee", redundant, LABELS)
    assert r["predicted"] == "scam" and r["top_words"] == [] and r["distributed"]


def test_occlusion_caps_words():
    r = occlusion(" ".join(["word"] * 100), fake_proba, LABELS, max_words=10)
    assert len(r["words"]) == 10 and r["truncated"]


def test_occlusion_works_when_saturated():
    def saturated(texts):
        # P(ham) = e^-40 with 'prize', e^-30 without: indistinguishable in float32 probs
        return np.array([[-40.0 if "prize" in t else -30.0, -50.0, 0.0] for t in texts])
    r = occlusion("claim your prize now", saturated, LABELS)
    assert r["top_words"][0]["word"] == "prize"
    assert r["top_words"][0]["importance"] == pytest.approx(10.0, rel=1e-3)


def test_occlusion_empty():
    assert occlusion("   ", fake_proba, LABELS)["words"] == []


def test_signals_detect_common_patterns():
    text = ("URGENT: your HBL account has been suspended. Verify your account within 24 hours. "
            "Click here and share your OTP.")
    keys = {s["key"] for s in find_signals(text)}
    assert {"urgency", "account_threat", "credentials", "impersonation", "link_prompt"} <= keys


def test_signals_quiet_on_normal_text():
    assert find_signals("Are we still on for dinner at 8? I'll bring dessert.") == []


def test_url_explainer_gives_reasons():
    from fraudshield.explain.url_shap import UrlExplainer
    from fraudshield.features.url_features import extract_frame
    from fraudshield.models.url_rf import make_pipeline

    benign = ["https://www.example.com/about", "https://news.site.org/world/today",
              "https://shop.store.com/item/123", "https://www.wiki.org/page"] * 5
    phish = ["http://10.0.0.5/login.php", "http://paypal.secure-verify.xyz/account",
             "http://apple.com-id-check.top/signin", "http://192.168.1.9/bank/verify"] * 5
    cfg = {"seed": 0, "random_forest": {"n_estimators": 20}}
    model = make_pipeline(cfg, max_depth=None, min_samples_leaf=1)
    model.fit(extract_frame(benign + phish), np.array([0] * 20 + [1] * 20))

    e = UrlExplainer(model)
    r = e.explain("http://10.0.0.7/login.php")
    assert r["score"] > 0.5
    assert r["reasons"] and all(x["contribution"] > 0 for x in r["reasons"])
    assert all(x["direction"] == "raises risk" for x in r["reasons"])
    r2 = e.explain("https://www.example.org/news")
    assert r2["score"] < 0.5
    assert all(x["contribution"] < 0 for x in r2["reasons"])
