import pytest

from fraudshield.inference.extract_urls import extract_urls, refang


def urls(text):
    return [f.url for f in extract_urls(text)]


@pytest.mark.parametrize("text,expected", [
    ("Visit https://example.com/path?x=1 now", ["https://example.com/path?x=1"]),
    ("go to www.daraz.pk today", ["http://www.daraz.pk"]),
    ("Click bit.ly/3abcXYZ to claim", ["http://bit.ly/3abcXYZ"]),
    ("Update at hbl-secure.top/login.", ["http://hbl-secure.top/login"]),
    ("login: http://192.168.4.20/bank/login.php", ["http://192.168.4.20/bank/login.php"]),
    ("admin panel 10.0.0.1:8080/admin", ["http://10.0.0.1:8080/admin"]),
    ("(see https://en.wikipedia.org/wiki/Phishing)", ["https://en.wikipedia.org/wiki/Phishing"]),
    ("https://en.wikipedia.org/wiki/Bracket_(disambiguation)",
     ["https://en.wikipedia.org/wiki/Bracket_(disambiguation)"]),
    ("Visit example.xyz now", ["http://example.xyz"]),
    ("Site: mybank.com.pk", ["http://mybank.com.pk"]),
])
def test_finds_urls(text, expected):
    assert urls(text) == expected


@pytest.mark.parametrize("text", [
    "Pay Rs.500 by 5.30pm",
    "e.g. this, i.e. that",
    "open report.pdf and notes.txt",
    "email me at ali@example.com",
    "Version 3.5.1 released",
    "Hi.How are you",
    "",
])
def test_ignores_non_urls(text):
    assert urls(text) == []


def test_defanged():
    found = extract_urls("Do not open hxxp://evil[.]com/login or bad(.)xyz/x")
    assert [f.url for f in found] == ["http://evil.com/login", "http://bad.xyz/x"]
    assert all(f.defanged for f in found)


def test_positions_and_flags():
    text = "Go to bit.ly/abc now"
    f = extract_urls(text)[0]
    assert text[f.start:f.end] == "bit.ly/abc"
    assert not f.scheme_given
    g = extract_urls("https://a.com")[0]
    assert g.scheme_given


def test_dedup_and_limit():
    text = " ".join(f"https://site{i}.com" for i in range(8)) + " https://site0.com/"
    found = extract_urls(text, limit=5)
    assert len(found) == 5
    assert found[0].url == "https://site0.com"


def test_refang():
    assert refang("hxxps://a[.]b[dot]com") == "https://a.b.com"
