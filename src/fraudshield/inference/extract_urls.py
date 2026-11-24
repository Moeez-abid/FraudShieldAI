"""Find URLs inside a pasted message.

Handles:
  * normal links         https://example.com/path, www.example.com
  * bare domains         bit.ly/3abc, hbl-secure.top/login, example.pk
  * defanged links       hxxp://evil[.]com, evil(.)com, evil[dot]com
  * raw IPs              http://192.168.1.10/login, 10.0.0.1:8080/admin

Avoids common false positives (Rs.500, e.g., file.txt, user@example.com e-mail
addresses). A bare domain with no scheme, www or path must use a common TLD.

Nothing is fetched or resolved: this is string matching only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tldextract

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

_DOT = r"(?:\.|\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\))"
_URL_CHARS = r"[^\s<>\"'`]"

_SCHEME_RE = re.compile(rf"(?i)\b(?:h[tx]{{2}}ps?|ftp)(?:://|\[://\]|\[:\]//){_URL_CHARS}+")
_IP_RE = re.compile(rf"(?<![\w.@])\d{{1,3}}(?:{_DOT}\d{{1,3}}){{3}}(?::\d{{2,5}})?(?:/{_URL_CHARS}*)?")
_BARE_RE = re.compile(
    rf"(?i)(?<![\w@.\-/])(?:www{_DOT})?(?:[a-z0-9](?:[a-z0-9-]{{0,61}}[a-z0-9])?{_DOT})+[a-z]{{2,24}}"
    rf"(?::\d{{2,5}})?(?:/{_URL_CHARS}*)?(?![\w@])"
)

# For bare host-only matches ("visit example.xyz"), require one of these TLDs or a
# 2-letter country code, so sentence typos like "Hi.How" (.how is a real gTLD) are skipped.
_COMMON_TLDS = {
    "com", "net", "org", "info", "biz", "xyz", "top", "online", "site", "club", "shop",
    "live", "icu", "vip", "app", "dev", "link", "click", "buzz", "store", "tech", "website",
    "space", "fun", "cyou", "rest", "lol", "work", "page", "gov", "edu", "mobi", "pro",
}

_TRAILING = ".,;:!?'\")]}>"


@dataclass(frozen=True)
class FoundUrl:
    original: str  # text exactly as it appeared in the message
    url: str       # refanged, with http:// added if no scheme was given
    start: int
    end: int
    scheme_given: bool
    defanged: bool


def refang(s: str) -> str:
    s = re.sub(r"(?i)^h[tx]{2}p", lambda m: "http" if m.group(0)[0].islower() else "HTTP", s)
    s = re.sub(r"(?i)\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\)", ".", s)
    s = s.replace("[://]", "://").replace("[:]//", "://")
    return s


def _trim(text: str, start: int, end: int) -> int:
    """Drop trailing punctuation, keeping a ')' only if the URL contains a matching '('."""
    while end > start and text[end - 1] in _TRAILING:
        if text[end - 1] == ")" and text[start:end].count("(") >= text[start:end].count(")"):
            break
        end -= 1
    return end


def _valid_host(candidate: str, has_path_or_www: bool) -> bool:
    host = re.split(r"[/:?#]", re.sub(r"(?i)^[a-z]+://", "", candidate), maxsplit=1)[0]
    ext = _EXTRACT(host)
    if not ext.domain or not ext.suffix:
        return False
    if has_path_or_www:
        return True
    last = ext.suffix.rsplit(".", 1)[-1].lower()
    return last in _COMMON_TLDS or len(last) == 2


def extract_urls(text: str, limit: int = 5) -> list[FoundUrl]:
    taken: list[tuple[int, int]] = []
    found: list[FoundUrl] = []

    def overlaps(s: int, e: int) -> bool:
        return any(s < te and ts < e for ts, te in taken)

    candidates = []
    for m in _SCHEME_RE.finditer(text):
        candidates.append((m.start(), _trim(text, m.start(), m.end()), "scheme"))
    for m in _IP_RE.finditer(text):
        candidates.append((m.start(), _trim(text, m.start(), m.end()), "ip"))
    for m in _BARE_RE.finditer(text):
        candidates.append((m.start(), _trim(text, m.start(), m.end()), "bare"))

    priority = {"scheme": 0, "ip": 1, "bare": 2}
    for start, end, kind in sorted(candidates, key=lambda c: (priority[c[2]], c[0])):
        if end <= start or overlaps(start, end):
            continue
        original = text[start:end]
        fixed = refang(original)
        if kind == "bare":
            has_path_or_www = "/" in fixed or fixed.lower().startswith("www.")
            if not _valid_host(fixed, has_path_or_www):
                continue
        taken.append((start, end))
        scheme_given = kind == "scheme"
        url = fixed if scheme_given else "http://" + fixed
        found.append(FoundUrl(original, url, start, end, scheme_given, fixed != original))

    found.sort(key=lambda f: f.start)
    unique, seen = [], set()
    for f in found:
        key = f.url.lower().rstrip("/")
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique[:limit]
