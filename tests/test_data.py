import pandas as pd

from fraudshield.data.clean_messages import dedup_key, normalise_text
from fraudshield.data.clean_urls import _normalise
from fraudshield.data.split import grouped_split, stratified_split

FRACS = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_normalise_text_keeps_case_and_unescapes():
    assert normalise_text("  URGENT!!  you &amp; me\n\n") == "URGENT!! you & me"
    assert normalise_text("bad�char") == "badchar"


def test_dedup_key_ignores_case_punct_digits():
    assert dedup_key("Call 0800-123 NOW!") == dedup_key("call 0900 456 now")
    assert dedup_key("hello there") != dedup_key("goodbye there")


def test_url_normalise_lowercases_host_only():
    assert _normalise("HTTPS://WWW.Example.COM/Path/File") == "https://www.example.com/Path/File"
    assert _normalise("Example.COM/AbC") == "example.com/AbC"
    assert _normalise("http://a.com/?x=1&amp;y=2") == "http://a.com/?x=1&y=2"


def test_stratified_split_sizes_and_disjoint():
    df = pd.DataFrame({"id": range(200), "label": ["a"] * 160 + ["b"] * 40})
    tr, va, te = stratified_split(df, "label", FRACS, seed=0)
    assert len(tr) + len(va) + len(te) == 200
    assert not (set(tr.id) & set(va.id) | set(tr.id) & set(te.id) | set(va.id) & set(te.id))
    assert abs(tr.label.eq("b").mean() - 0.2) < 0.02


def test_grouped_split_has_no_domain_overlap():
    df = pd.DataFrame({
        "id": range(1000),
        "reg_domain": [f"d{i % 120}.com" for i in range(1000)],
        "label": [i % 2 for i in range(1000)],
    })
    tr, va, te = grouped_split(df, "reg_domain", FRACS, seed=0)
    assert len(tr) + len(va) + len(te) == 1000
    assert not set(tr.reg_domain) & set(va.reg_domain)
    assert not set(tr.reg_domain) & set(te.reg_domain)
    assert not set(va.reg_domain) & set(te.reg_domain)


def test_challenge_agreement_and_kappa():
    from fraudshield.data.challenge import agreement
    df = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "label_moeez": ["scam", "ham", "spam", "scam"],
        "label_ijtaba": ["scam", "ham", "scam", "scam"],
        "label_final": ["scam", "ham", None, "Scam "],
    })
    r = agreement(df, "messages")
    assert r["both_labelled"] == 4 and r["final_labelled"] == 3
    assert r["raw_agreement"] == 0.75 and r["disagreements"] == ["c"]
    assert -1 <= r["cohen_kappa"] <= 1 and r["invalid_labels"] == {}
    bad = agreement(df.assign(label_final=["fraud", "ham", None, None]), "messages")
    assert bad["invalid_labels"] == {"label_final": ["fraud"]}
