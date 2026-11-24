import pytest

from fraudshield.evaluation.metrics import binary_metrics, message_metrics

LABELS = ["ham", "spam", "scam"]


def test_message_metrics_perfect():
    y = ["ham", "spam", "scam", "ham"]
    m = message_metrics(y, y, LABELS)
    assert m["macro_f1"] == 1.0
    assert m["fpr_benign_flagged"] == 0.0
    assert m["slices"]["scam_vs_ham"]["recall_flagged"] == 1.0


def test_message_metrics_scam_called_spam_still_flagged():
    y_true = ["ham", "scam", "scam"]
    y_pred = ["ham", "spam", "scam"]
    s = message_metrics(y_true, y_pred, LABELS)["slices"]["scam_vs_ham"]
    assert s["recall_flagged"] == 1.0
    assert s["recall_exact_class"] == 0.5


def test_message_metrics_fpr():
    m = message_metrics(["ham", "ham", "ham", "ham"], ["ham", "spam", "ham", "ham"], LABELS)
    assert m["fpr_benign_flagged"] == 0.25


def test_binary_metrics():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.4])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["fpr"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.75)
