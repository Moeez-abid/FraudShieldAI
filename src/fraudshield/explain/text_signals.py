"""Rule-based scam signal tags for messages.

These are HEURISTICS, not model reasoning. The app shows them under a separate
"Common scam signals found" heading so users don't mistake them for why the model
decided. Each match returns the matched text so it can be highlighted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    key: str
    title: str
    explanation: str
    pattern: re.Pattern


def _p(*alternatives: str) -> re.Pattern:
    return re.compile(r"(?i)\b(?:" + "|".join(alternatives) + r")\b")


SIGNALS: tuple[Signal, ...] = (
    Signal("urgency", "Urgency or deadline",
           "Scams push you to act before you can think or check.",
           _p(r"act now", r"urgent(?:ly)?", r"immediately", r"right away", r"asap",
              r"within \d+ ?(?:hours?|hrs?|minutes?|mins?|days?)", r"today only", r"last chance",
              r"expires? (?:today|soon|tonight)", r"final (?:notice|warning|reminder)",
              r"limited time")),
    Signal("credentials", "Asks for login details or codes",
           "Real organisations never ask for your password, PIN or one-time code by message.",
           _p(r"verify your (?:account|identity|details)", r"confirm your (?:account|identity|details)",
              r"(?:share|send|enter|provide) (?:your |the )?(?:otp|pin|password|passcode|cvv|code)",
              r"otp", r"one[- ]time (?:password|code|pin)", r"login details", r"log ?in to (?:verify|confirm|restore)",
              r"update your (?:details|information|account)", r"security code")),
    Signal("account_threat", "Threat to your account",
           "Claims your account is locked or suspended are a common scare tactic.",
           _p(r"(?:account|card|sim|number) (?:has been |will be |is )?(?:suspended|blocked|locked|deactivated|closed|restricted)",
              r"unusual (?:activity|sign[- ]?in)", r"unauthori[sz]ed (?:access|transaction|login)",
              r"suspicious activity")),
    Signal("prize", "Prize, reward or free money",
           "Unexpected winnings that need a fee or your details are a classic scam.",
           _p(r"you(?:'ve| have)? won", r"winner", r"congratulations", r"claim (?:your )?(?:prize|reward|gift)",
              r"free (?:gift|iphone|voucher|money)", r"cash prize", r"lottery", r"jackpot", r"selected for")),
    Signal("payment", "Payment or money request",
           "Requests to pay fees, send money or buy gift cards are high-risk.",
           _p(r"(?:pay|send|transfer) (?:a |the )?(?:fee|amount|money|payment)", r"processing fee",
              r"gift ?cards?", r"easypaisa", r"jazzcash", r"bitcoin|crypto", r"wire transfer",
              r"customs (?:fee|duty|charge)", r"outstanding (?:balance|payment|amount)")),
    Signal("impersonation", "Claims to be a known organisation",
           "Scammers pose as banks, couriers or government offices. Check through official channels.",
           _p(r"bank", r"hbl", r"ubl", r"meezan", r"paypal", r"amazon", r"apple", r"microsoft", r"netflix",
              r"dhl", r"fedex", r"tcs", r"leopards", r"pakistan post", r"fbr", r"nadra", r"pta",
              r"irs", r"hmrc", r"customs", r"police")),
    Signal("link_prompt", "Pushes you to click a link",
           "Links in unexpected messages can lead to fake login pages.",
           _p(r"click (?:here|the link|below|on the link)", r"tap (?:here|the link)", r"visit (?:the )?link",
              r"follow the link", r"open the link")),
)


def find_signals(text: str) -> list[dict]:
    """Return one entry per signal type found, with the distinct matched phrases."""
    found = []
    for sig in SIGNALS:
        matches = [m.group(0) for m in sig.pattern.finditer(text)]
        if matches:
            seen = list(dict.fromkeys(m.lower() for m in matches))
            found.append({"key": sig.key, "title": sig.title, "explanation": sig.explanation,
                          "matches": seen})
    return found
