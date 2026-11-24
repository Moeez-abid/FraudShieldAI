"""Per-URL explanations from the Random Forest using SHAP TreeExplainer.

Turns the top SHAP contributions into short plain-English reasons. Only features
that push toward the predicted direction are shown; the raw values are included
so the UI can display them.
"""

from __future__ import annotations

import numpy as np
import shap
from sklearn.pipeline import Pipeline

from fraudshield.features.url_features import FEATURE_DESCRIPTIONS, extract_frame

# Boolean features: (text when present, text when absent or None to skip).
_BOOL = {
    "is_ip_host": ("Uses a raw IP address instead of a domain name", "Uses a normal domain name"),
    "is_punycode": ("Uses punycode (xn--), which can disguise look-alike letters", None),
    "has_port": ("Specifies an unusual port", None),
    "is_https": ("Uses HTTPS", "Doesn't use HTTPS"),
    "has_scheme": ("Includes http(s)://", None),
    "is_shortener": ("Uses a link shortener that hides the real destination", None),
    "brand_outside_domain": ("Mentions a well-known brand that isn't the site's real domain", None),
    "has_double_slash_path": ("Contains '//' in the path (possible redirect)", None),
}
# Numeric features: neutral wording with the value. The UI shows the direction
# (raises / lowers risk) from the contribution sign, so the text never has to guess.
_NUM = {
    "n_suspicious_words": "Words common in phishing links: {v:.0f}",
    "n_subdomains": "Subdomain levels: {v:.0f}",
    "n_hyphens": "Hyphens: {v:.0f}",
    "n_dots": "Dots: {v:.0f}",
    "n_at": "'@' characters: {v:.0f}",
    "host_entropy": "Host-name randomness: {v:.2f}",
    "path_entropy": "Path randomness: {v:.2f}",
    "digit_ratio": "Digits make up {v:.0%} of characters",
    "letter_ratio": "Letters make up {v:.0%} of characters",
    "longest_digit_run": "Longest run of digits: {v:.0f}",
    "url_len": "URL length: {v:.0f} characters",
    "host_len": "Host-name length: {v:.0f} characters",
    "path_len": "Path length: {v:.0f} characters",
    "query_len": "Query-string length: {v:.0f} characters",
    "n_path_segments": "Path segments: {v:.0f}",
    "tld_len": "Top-level-domain length: {v:.0f}",
}


def _sentence(name: str, value: float, toward_phishing: bool) -> str | None:
    if name.startswith("tld_"):
        if value < 0.5:  # one-hot 0: "isn't .com" is not a useful reason
            return None
        tld = name[4:]
        if tld.startswith("infrequent"):
            return "Uncommon top-level domain"
        return f"Top-level domain: .{tld}"
    if name in _BOOL:
        present, absent = _BOOL[name]
        return present if value >= 0.5 else absent
    template = _NUM.get(name)
    if template:
        return template.format(v=value)
    return f"{FEATURE_DESCRIPTIONS.get(name, name)}: {value:g}"


class UrlExplainer:
    def __init__(self, model: Pipeline):
        self.model = model
        self.pre = model.named_steps["pre"]
        self.rf = model.named_steps["rf"]
        self.names = [n.split("__", 1)[-1] for n in self.pre.get_feature_names_out()]
        self.explainer = shap.TreeExplainer(self.rf)

    def explain(self, url: str, top_k: int = 4) -> dict:
        X = self.pre.transform(extract_frame([url]))
        score = float(self.rf.predict_proba(X)[0, 1])
        sv = self.explainer.shap_values(X, check_additivity=False)
        # shap returns [class0, class1] (older) or (n, features, classes) (newer)
        contrib = np.asarray(sv[1] if isinstance(sv, list) else sv[..., 1])[0]
        toward_phishing = score >= 0.5
        order = np.argsort(contrib)[::-1] if toward_phishing else np.argsort(contrib)
        reasons = []
        for i in order:
            c = float(contrib[i])
            if (c <= 0) if toward_phishing else (c >= 0):
                break
            text = _sentence(self.names[i], float(X[0, i]), c > 0)
            if text:
                reasons.append({"feature": self.names[i], "value": float(X[0, i]),
                                "contribution": c, "text": text,
                                "direction": "raises risk" if c > 0 else "lowers risk"})
            if len(reasons) == top_k:
                break
        return {"url": url, "score": score, "reasons": reasons}
