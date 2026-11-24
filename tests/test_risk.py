import numpy as np

from fraudshield.inference.risk import band, choose_thresholds, max_band


def test_thresholds_meet_targets():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(1000)].astype(int)
    score = np.r_[rng.beta(2, 5, 1000), rng.beta(5, 2, 1000)]
    t = choose_thresholds(y, score, high_precision=0.9, medium_max_benign_flagged=0.1)
    assert t["medium"] <= t["high"]
    assert t["at_high"]["precision"] >= 0.9
    assert t["at_medium"]["benign_flagged_rate"] <= 0.1


def test_band_edges():
    t = {"medium": 0.3, "high": 0.7}
    assert band(0.29, t) == "Low"
    assert band(0.30, t) == "Medium"
    assert band(0.70, t) == "High"


def test_max_band():
    assert max_band("Low", "High", "Medium") == "High"
    assert max_band(None, "Medium") == "Medium"
    assert max_band() == "Low"
