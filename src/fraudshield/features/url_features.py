"""Lexical and structural URL features.

Pure string parsing: nothing here resolves DNS, fetches pages or contacts any host.
Every feature has a plain-English description (FEATURE_DESCRIPTIONS) used by the
explanation layer.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from functools import lru_cache
from urllib.parse import urlsplit

import pandas as pd
import tldextract

# Bundled public-suffix snapshot only: deterministic and offline.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

SUSPICIOUS_WORDS = (
    "login", "log-in", "signin", "sign-in", "verify", "verification", "secure", "account",
    "update", "confirm", "banking", "bank", "password", "credential", "wallet", "free",
    "bonus", "prize", "winner", "gift", "unlock", "suspend", "billing", "invoice", "webscr",
)
BRANDS = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "amazon", "netflix",
    "google", "facebook", "instagram", "whatsapp", "chase", "wellsfargo", "hsbc", "barclays",
    "dhl", "fedex", "usps", "irs", "hmrc", "ebay", "linkedin", "adobe", "dropbox",
    "binance", "coinbase", "metamask", "steam", "spotify",
)
SHORTENERS = frozenset({
    "bit.ly", "bitly.com", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "s.id", "t.ly", "v.gd",
})

FEATURE_DESCRIPTIONS = {
    "url_len": "Overall URL length",
    "host_len": "Length of the host name",
    "path_len": "Length of the path",
    "query_len": "Length of the query string",
    "n_path_segments": "Number of path segments",
    "n_dots": "Number of dots",
    "n_hyphens": "Number of hyphens",
    "n_underscores": "Number of underscores",
    "n_at": "Contains '@' (can hide the real destination)",
    "n_qmarks": "Number of '?' characters",
    "n_equals": "Number of '=' characters",
    "n_amps": "Number of '&' characters",
    "n_percent": "Number of percent-encoded characters",
    "n_tilde": "Number of '~' characters",
    "digit_ratio": "Share of characters that are digits",
    "letter_ratio": "Share of characters that are letters",
    "n_non_ascii": "Number of non-ASCII characters",
    "is_ip_host": "Uses a raw IP address instead of a domain name",
    "n_subdomains": "Number of subdomain levels",
    "is_punycode": "Uses punycode (xn--), which can disguise look-alike characters",
    "has_port": "Specifies an explicit port",
    "is_https": "Uses HTTPS",
    "has_scheme": "Includes a scheme (http/https)",
    "n_suspicious_words": "Number of words often used in phishing (login, verify, account...)",
    "brand_outside_domain": "Mentions a well-known brand outside the actual registered domain",
    "has_double_slash_path": "Contains '//' inside the path (possible redirect)",
    "host_entropy": "Randomness of the host name",
    "path_entropy": "Randomness of the path",
    "longest_digit_run": "Longest run of consecutive digits",
    "is_shortener": "Uses a link-shortening service that hides the destination",
    "tld_len": "Length of the top-level domain",
}
NUMERIC_FEATURES = list(FEATURE_DESCRIPTIONS)
CATEGORICAL_FEATURES = ["tld"]

_DIGIT_RUN = re.compile(r"\d+")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


@lru_cache(maxsize=200_000)
def registered_domain(url: str) -> str:
    """eTLD+1 (e.g. 'example.co.uk'); falls back to the host for IPs/unknown suffixes."""
    host = _split(url)[1]
    ext = _EXTRACT(host)
    return ext.top_domain_under_public_suffix if hasattr(ext, "top_domain_under_public_suffix") \
        else (ext.registered_domain or host)


def _split(url: str):
    raw = url.strip()
    has_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw))
    try:
        parts = urlsplit(raw if has_scheme else "http://" + raw)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:  # malformed netloc / port
        parts, host, port = urlsplit("http://invalid"), "", None
    return parts, host, port, has_scheme


def extract(url: str) -> dict:
    parts, host, port, has_scheme = _split(url)
    ext = _EXTRACT(host)
    reg_label = ext.domain.lower()
    path, query = parts.path or "", parts.query or ""
    full = url.strip()
    lower = full.lower()
    n = max(1, len(full))
    outside = lower.replace(reg_label, "", 1) if reg_label else lower

    return {
        "url_len": len(full),
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(query),
        "n_path_segments": len([p for p in path.split("/") if p]),
        "n_dots": full.count("."),
        "n_hyphens": full.count("-"),
        "n_underscores": full.count("_"),
        "n_at": full.count("@"),
        "n_qmarks": full.count("?"),
        "n_equals": full.count("="),
        "n_amps": full.count("&"),
        "n_percent": full.count("%"),
        "n_tilde": full.count("~"),
        "digit_ratio": sum(c.isdigit() for c in full) / n,
        "letter_ratio": sum(c.isalpha() for c in full) / n,
        "n_non_ascii": sum(ord(c) > 127 for c in full),
        "is_ip_host": int(_is_ip(host)),
        "n_subdomains": len([s for s in ext.subdomain.split(".") if s]) if ext.subdomain else 0,
        "is_punycode": int("xn--" in host),
        "has_port": int(port is not None),
        "is_https": int(parts.scheme == "https"),
        "has_scheme": int(has_scheme),
        "n_suspicious_words": sum(w in lower for w in SUSPICIOUS_WORDS),
        "brand_outside_domain": int(any(b in outside and b != reg_label for b in BRANDS)),
        "has_double_slash_path": int("//" in path),
        "host_entropy": _entropy(host),
        "path_entropy": _entropy(path),
        "longest_digit_run": max((len(m) for m in _DIGIT_RUN.findall(full)), default=0),
        "is_shortener": int(registered_domain(url) in SHORTENERS),
        "tld_len": len(ext.suffix),
        "tld": ext.suffix.lower() or ("<ip>" if _is_ip(host) else "<none>"),
    }


def extract_frame(urls) -> pd.DataFrame:
    return pd.DataFrame([extract(u) for u in urls])
