"""Optional user context: rules that can only RAISE the risk band.

Many scams look harmless as text ("Hi mum, this is my new number", "Your parcel is
held") and only give themselves away through context the user knows: who sent it,
whether they were expecting it, whether they even have an account with the named
organisation. The user can answer three optional questions; these rules then set a
minimum band ("floor") with a plain-language reason.

Design rules:
- Context never goes into a model. The frozen models see exactly the same input.
- Context can only raise the rating, never lower it. A saved contact can be hacked and
  a sender name can be spoofed, so "it looks official" is never a reason to relax.
- With no context given, nothing changes: the report is identical to the evaluated
  system. These rules are NOT evaluated on the test set (it has no context).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fraudshield.explain.text_signals import SIGNALS
from fraudshield.inference.risk import max_band

# Answer values (the app maps its radio options to these).
SENDER_CONTACT = "saved_contact"     # someone in your contacts
SENDER_UNKNOWN = "unknown_number"    # an unknown mobile number, e.g. 03xx...
SENDER_OFFICIAL = "official_sender"  # a sender name or short code, e.g. "HBL", 8558
SENDERS = (SENDER_CONTACT, SENDER_UNKNOWN, SENDER_OFFICIAL)
YES, NO = "yes", "no"


@dataclass(frozen=True)
class Context:
    sender: str | None = None       # one of SENDERS, or None = not sure / not given
    expected: str | None = None     # YES / NO / None: were you expecting this message?
    has_account: str | None = None  # YES / NO / None: account or order with the named organisation?

    def __post_init__(self):
        if self.sender not in (None, *SENDERS):
            raise ValueError(f"unknown sender: {self.sender!r}")
        for name in ("expected", "has_account"):
            if getattr(self, name) not in (None, YES, NO):
                raise ValueError(f"{name} must be 'yes', 'no' or None")

    @property
    def empty(self) -> bool:
        return self.sender is None and self.expected is None and self.has_account is None


def _p(*alternatives: str) -> re.Pattern:
    return re.compile(r"(?i)\b(?:" + "|".join(alternatives) + r")\b")


_SIG = {s.key: s.pattern for s in SIGNALS}

# Claims to be an organisation: the impersonation signal plus wallets and aid schemes.
_ORG = _p(r"easypaisa", r"jazz ?cash", r"sadapay", r"nayapay", r"bisp", r"ehsaas",
          r"benazir", r"daraz", r"jazz", r"zong", r"telenor", r"ufone", r"wapda", r"k-?electric")

# Claims to be family or a friend writing from a new number.
_FAMILY = _p(r"(?:my )?new number", r"changed my number", r"my (?:old )?(?:phone|mobile|number) (?:is |was |got )?(?:broken|lost|stolen|not working|dead)",
             r"lost my phone", r"(?:hi|hello|salam|assalam o alaikum) (?:mum|mom|mummy|dad|papa|ammi|ami|abbu|abu|beta|bhai|baji|api)",
             r"it'?s me", r"this is your (?:son|daughter|brother|sister|friend|cousin|uncle|aunt|nephew|niece)")

# Topics where "I wasn't expecting this" matters.
_TRANSACTIONAL = _p(r"parcel", r"package", r"deliver\w*", r"shipment", r"shipped", r"courier",
                    r"order", r"payment", r"paid", r"transaction", r"refund", r"prize", r"won",
                    r"winner", r"reward", r"lucky draw", r"cash ?back", r"transfer\w*",
                    r"deposit\w*", r"rs\.?", r"pkr", r"amount", r"invoice", r"bill")

# Asks you to SEND or SHARE a code. Negated warnings ("do not share your OTP", "never
# share this code") are excluded, so genuine OTP messages don't match.
_CODE_REQUEST = re.compile(
    r"(?i)(?<!not )(?<!never )(?<!n't )\b(?:send|share|forward|tell|give|provide|reply with|text)\b"
    r"(?:\s+\w+){0,4}?\s+(?:otp|pin|password|passcode|cvv|(?:verification |security |\d[- ]digit )?code)\b")
_MONEY_REQUEST = _p(r"(?:send|transfer|lend|need|pay)\b(?:\s+\w+){0,4}?\s+(?:money|rs\.?|pkr|amount|\d[\d,]{2,})",
                    r"easypaisa", r"jazz ?cash", r"urgent(?:ly)? need")


def _has(pattern: re.Pattern, text: str) -> bool:
    return bool(pattern.search(text))


def context_floor(text: str, context: Context | None, has_urls: bool = False) -> tuple[str | None, list[str]]:
    """Return (minimum band or None, reasons). None means the context changes nothing."""
    if context is None or context.empty:
        return None, []

    org = _has(_SIG["impersonation"], text) or _has(_ORG, text)
    asks_code = _has(_CODE_REQUEST, text)
    asks_money = _has(_SIG["payment"], text) or _has(_MONEY_REQUEST, text)
    pressure = (asks_code or asks_money or has_urls or _has(_SIG["credentials"], text)
                or _has(_SIG["account_threat"], text) or _has(_SIG["link_prompt"], text))

    floor, notes = None, []

    def raise_to(b: str, note: str) -> None:
        nonlocal floor
        floor = max_band(floor, b)
        notes.append(note)

    if asks_code:
        raise_to("High", "It asks you to send or share a code. No real organisation or friend needs "
                         "your one-time code: sharing it hands over your account.")

    if context.sender == SENDER_UNKNOWN and org:
        raise_to("High", "It claims to be an organisation but came from an ordinary mobile number. "
                         "Banks, wallets and couriers don't send messages from personal numbers.")

    if context.sender == SENDER_UNKNOWN and _has(_FAMILY, text):
        if asks_money or asks_code:
            raise_to("High", "Someone claiming to be family or a friend on a new number is asking for "
                             "money or a code. Call them on their old number first.")
        else:
            raise_to("Medium", "Someone claiming to be family or a friend on a new number is a common "
                               "scam opener. Check by calling their old number.")

    if context.expected == NO and _has(_TRANSACTIONAL, text):
        if pressure:
            raise_to("High", "You weren't expecting this, and it asks you to act (pay, click, log in "
                             "or share details). Unexpected parcel, payment and prize messages that "
                             "ask for action are a common scam.")
        else:
            raise_to("Medium", "You weren't expecting this delivery, payment or prize message. "
                               "Check through the official app or website before acting.")

    if context.has_account == NO and org:
        if pressure:
            raise_to("High", "It's about an account or order you don't have, and it asks you to act. "
                             "That points to a scam.")
        else:
            raise_to("Medium", "It mentions an organisation you don't have an account with.")

    return floor, notes
