import pytest

from fraudshield.features.url_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    extract,
    extract_frame,
    registered_domain,
)


def test_all_features_present_and_numeric():
    f = extract("https://www.example.com/a/b?x=1")
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(f)
    assert all(isinstance(f[k], (int, float)) for k in NUMERIC_FEATURES)


@pytest.mark.parametrize("url,expected", [
    ("https://www.google.com", "google.com"),
    ("http://login.bbc.co.uk/path", "bbc.co.uk"),
    ("paypal.verify-account.xyz/webscr", "verify-account.xyz"),
    ("http://10.0.0.1/admin", "10.0.0.1"),
])
def test_registered_domain(url, expected):
    assert registered_domain(url) == expected


def test_ip_and_port():
    f = extract("http://192.168.1.10:8080/login")
    assert f["is_ip_host"] == 1 and f["has_port"] == 1 and f["tld"] == "<ip>"


def test_brand_outside_domain():
    assert extract("http://paypal.secure-check.com/login")["brand_outside_domain"] == 1
    assert extract("https://www.paypal.com/signin")["brand_outside_domain"] == 0


def test_shortener_and_punycode():
    assert extract("https://bit.ly/abc")["is_shortener"] == 1
    assert extract("http://xn--pple-43d.com")["is_punycode"] == 1


def test_scheme_flags():
    assert extract("example.com")["has_scheme"] == 0
    f = extract("https://example.com")
    assert f["has_scheme"] == 1 and f["is_https"] == 1


def test_malformed_url_does_not_raise():
    f = extract("http://[::1:bad/path")
    assert f["host_len"] == 0


def test_extract_frame_shape():
    df = extract_frame(["a.com", "https://b.org/x"])
    assert df.shape == (2, len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES))
