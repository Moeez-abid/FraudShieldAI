"""Cross-message rules: scam patterns that only exist across a conversation.

Windows (conversation.py) let the model read a few neighbouring messages together, but
an opener and the request it sets up can be many messages apart ("new number" in
message 1, "send money" in message 9). These hand-written rules look at the whole chat,
whatever its length. Like the other app rules they only RAISE the rating, they never
change model inputs, and they are not evaluated on labelled conversations (none
exist); their trigger rate on benign synthetic chats is reported by
conversation_thresholds.py.
"""

from __future__ import annotations

import re

from fraudshield.inference.context_rules import _CODE_REQUEST, _FAMILY, _MONEY_REQUEST
from fraudshield.inference.conversation import Turn
from fraudshield.inference.risk import max_band


def _p(*alternatives: str) -> re.Pattern:
    return re.compile(r"(?i)\b(?:" + "|".join(alternatives) + r")\b")


_WRONG_NUMBER = _p(r"wrong number", r"sorry,? (?:i|who) (?:\w+ ){0,4}(?:wrong|mistake)", r"is this (?:mr\.? |ms\.? )?\w+\?",
                   r"do i know you", r"sorry to disturb", r"i think i have the wrong")
_INVESTMENT = _p(r"crypto\w*", r"bitcoin", r"usdt", r"btc", r"trading", r"forex", r"invest\w*", r"mining",
                 r"binance", r"profits?", r"returns", r"signals? group", r"platform")
_GUARANTEE = _p(r"guaranteed (?:profits?|returns?|income)", r"\d{2,4} ?% (?:profit|returns?|in a (?:week|month|day))",
                r"(?:made|make|earn(?:ed)?) \d{2,4} ?%", r"double your money", r"risk[- ]free", r"no risk")
_CODE_MISTAKE = _p(r"(?:sent|send|forwarded) (?:you )?(?:a |the |my |an )?(?:\w+ ){0,3}code (?:to you )?by mistake",
                   r"code (?:came|went|was sent) to (?:you|your number)", r"by mistake (?:a |the )?(?:\w+ ){0,2}code")
_JOB = _p(r"part[- ]time", r"work from home", r"online (?:job|work|tasks?)", r"daily (?:income|earning|salary)",
          r"earn (?:rs\.?|pkr|\$|money|\d+)", r"like (?:videos|posts)", r"(?:review|rating|simple) tasks?",
          r"commission", r"hiring")
_FEE = _p(r"deposit\w*", r"registration fee", r"activation fee", r"(?:pay|send) (?:a |the )?(?:\w+ )?(?:fee|deposit)",
          r"recharge", r"top ?up", r"invest\w*", r"upgrade (?:your )?(?:account|level|vip)")
_ASK_MONEY = _MONEY_REQUEST


def _first(pattern: re.Pattern, turns: list[Turn]) -> Turn | None:
    return next((t for t in turns if pattern.search(t.text)), None)


def _later(pattern: re.Pattern, turns: list[Turn], after: int) -> Turn | None:
    return next((t for t in turns if t.index >= after and pattern.search(t.text)), None)


def conversation_floor(turns: list[Turn]) -> tuple[str | None, list[str]]:
    """turns = the OTHER person's messages, in order. Returns (minimum band or None, reasons)."""
    if len(turns) < 2:  # a single message: nothing cross-message to find
        return None, []
    floor, notes = None, []

    def raise_to(b: str, note: str) -> None:
        nonlocal floor
        floor = max_band(floor, b)
        notes.append(note)

    opener = _first(_FAMILY, turns)
    if opener:
        ask = _later(_CODE_REQUEST, turns, opener.index) or _later(_ASK_MONEY, turns, opener.index)
        if ask:
            raise_to("High", f"Message {opener.index} says it's a new number or a lost phone; message "
                             f"{ask.index} asks for money or a code. This is the classic 'family on a new "
                             f"number' scam. Call the person on their old number before doing anything.")

    wrong = _first(_WRONG_NUMBER, turns)
    if wrong:
        invest = _later(_INVESTMENT, turns, wrong.index + 1)
        if invest:
            ask = _later(_ASK_MONEY, turns, invest.index) or _later(_FEE, turns, invest.index)
            raise_to("High" if ask else "Medium",
                     f"Message {wrong.index} starts as a 'wrong number' chat; message {invest.index} turns to "
                     f"investment or crypto" + (f" and message {ask.index} asks you to pay or invest" if ask else "")
                     + ". Friendly strangers who move to investments are a common scam.")

    pitch = _first(_INVESTMENT, turns)
    if pitch and not wrong:
        ask = _later(_ASK_MONEY, turns, pitch.index) or _later(_FEE, turns, pitch.index + 1)
        if ask:
            promise = _first(_GUARANTEE, turns)
            raise_to("High" if promise else "Medium",
                     f"Message {pitch.index} pitches crypto, trading or an investment and message {ask.index} "
                     f"asks you to send money" + (f"; message {promise.index} promises guaranteed or very "
                     f"high profits, which real investments never do" if promise else "")
                     + ". Check with someone you trust before paying anything.")

    mistake = _first(_CODE_MISTAKE, turns)
    if mistake:
        ask = _later(_CODE_REQUEST, turns, mistake.index)
        if ask:
            raise_to("High", f"Message {mistake.index} says a code was sent to you by mistake; message "
                             f"{ask.index} asks you to send it. That code is for YOUR account (often "
                             f"WhatsApp): sending it hands your account over.")

    job = _first(_JOB, turns)
    if job:
        fee = _later(_FEE, turns, job.index + 1)
        if fee:
            raise_to("High", f"Message {job.index} offers easy work or income; message {fee.index} asks for a "
                             f"deposit, fee or top-up. Real jobs don't make you pay to start.")

    asks = [t for t in turns if _CODE_REQUEST.search(t.text) or _ASK_MONEY.search(t.text)]
    if len(asks) >= 2:
        raise_to("Medium", f"Messages {', '.join(str(t.index) for t in asks[:5])} keep asking for money or a "
                           f"code. Repeated pressure is a warning sign.")
    return floor, notes
