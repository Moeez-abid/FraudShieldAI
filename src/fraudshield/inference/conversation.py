"""Conversations: parse a pasted chat into turns and cut it into model-sized pieces.

The message models were trained on single SMS (median 20 tokens) and DistilBERT reads
at most 128 tokens. Joining a whole chat into one text fails twice: the end of a long
chat is cut off, and a scam at the end is diluted by friendly small talk before it.
Instead, the chat is scored as overlapping **windows**:

- every run of 1..max_messages consecutive messages from the other person that fits in
  the token budget (short messages -> windows of several; long ones -> windows of one);
- a single message longer than the budget is cut into overlapping chunks (stride = half
  the budget), so nothing is skipped and a sentence on a boundary appears whole once.

The conversation score is the highest window score. Its thresholds are tuned separately
(conversation_thresholds.py), because the maximum over many windows flags benign chats
more often than one message does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

MAX_TURNS = 60

# WhatsApp exports. iOS: "[24/11/2026, 10:15:32] Ali: hi"  Android: "24/11/2026, 10:15 - Ali: hi"
_DATE = r"\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}"
_TIME = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap]\.?[Mm]\.?)?"
_WA_IOS = re.compile(rf"^\[({_DATE}),?\s+({_TIME})\]\s+([^:]{{1,40}}?):\s?(.*)$")
_WA_ANDROID = re.compile(rf"^({_DATE}),?\s+({_TIME})\s+[-–]\s+([^:]{{1,40}}?):\s?(.*)$")
# Plain "Name: message" lines (name of 1-4 words, no digits-only names, no URLs like http:).
_NAMED = re.compile(r"^([A-Za-z؀-ۿ][\w.'\-]*(?: [\w.'\-]+){0,3}):\s+(.+)$")
_SYSTEM = re.compile(r"(?i)end-to-end encrypted|<media omitted>|(?:image|video|audio|sticker|document) omitted|"
                     r"this message was deleted|missed (?:voice|video) call")
_INVISIBLE = dict.fromkeys(map(ord, "‎‏‪‬﻿"), None)
ME_NAMES = frozenset({"you", "me", "myself"})


@dataclass(frozen=True)
class Turn:
    index: int             # 1-based position in the conversation
    speaker: str | None    # None when the paste has no names
    text: str


@dataclass(frozen=True)
class Piece:
    """One model input: consecutive messages (start..end turn index) or a chunk of one."""
    start: int
    end: int
    text: str
    n_messages: int
    part: tuple[int, int] | None = None  # (k, n) when a long message was cut into n chunks


def parse_conversation(text: str) -> tuple[list[Turn], str]:
    """Split pasted chat text into turns. Returns (turns, format name)."""
    lines = [ln.translate(_INVISIBLE).replace(" ", " ").rstrip() for ln in (text or "").splitlines()]
    lines_nonempty = [ln for ln in lines if ln.strip()]
    if not lines_nonempty:
        return [], "empty"

    for fmt, pattern in (("whatsapp", _WA_IOS), ("whatsapp", _WA_ANDROID)):
        if sum(bool(pattern.match(ln.strip())) for ln in lines_nonempty) >= max(1, len(lines_nonempty) // 3):
            return _parse_with(lines, lambda ln: _wa(pattern, ln)), fmt

    named = [_NAMED.match(ln.strip()) for ln in lines_nonempty]
    speakers = {m.group(1).strip() for m in named if m}
    if sum(map(bool, named)) >= 2 and 1 <= len(speakers) <= 6 and sum(map(bool, named)) >= len(lines_nonempty) // 2:
        return _parse_with(lines, lambda ln: _plain(ln)), "named"

    # No names: blank-line separated blocks if there are any, else one message per line.
    if any(not ln.strip() for ln in lines) and len(lines_nonempty) > 1:
        blocks = re.split(r"\n\s*\n", "\n".join(lines).strip())
        msgs = [" ".join(b.split()) for b in blocks if b.strip()]
        fmt = "blocks"
    else:
        msgs = [" ".join(ln.split()) for ln in lines_nonempty]
        fmt = "lines"
    msgs = [m for m in msgs if not _SYSTEM.search(m)][:MAX_TURNS]
    return [Turn(i + 1, None, m) for i, m in enumerate(msgs)], fmt


def _wa(pattern: re.Pattern, line: str):
    m = pattern.match(line.strip())
    return (m.group(3).strip(), m.group(4)) if m else None


def _plain(line: str):
    m = _NAMED.match(line.strip())
    return (m.group(1).strip(), m.group(2)) if m else None


def _parse_with(lines: list[str], match: Callable) -> list[Turn]:
    raw: list[list] = []
    for ln in lines:
        if not ln.strip():
            continue
        hit = match(ln)
        if hit:
            raw.append([hit[0], hit[1].strip()])
        elif raw:  # continuation of a multi-line message
            raw[-1][1] += " " + ln.strip()
    turns = []
    for speaker, msg in raw:
        msg = " ".join(msg.split())
        if msg and not _SYSTEM.search(msg):
            turns.append(Turn(len(turns) + 1, speaker, msg))
        if len(turns) >= MAX_TURNS:
            break
    return turns


def speakers(turns: list[Turn]) -> list[str]:
    return list(dict.fromkeys(t.speaker for t in turns if t.speaker))


def guess_me(turns: list[Turn]) -> str | None:
    for s in speakers(turns):
        if s.lower() in ME_NAMES:
            return s
    return None


# --------------------------------------------------------------------------- tokens
def word_token_counter(model) -> Callable[[list[str]], list[int]]:
    """Tokens per whitespace word. Exact for BERT-style tokenizers (they split on
    whitespace before word pieces); an estimate (1 per ~4 chars) otherwise."""
    tok = getattr(model, "tok", None)
    if tok is not None:
        return lambda words: [len(ids) for ids in tok(words, add_special_tokens=False)["input_ids"]] if words else []
    return lambda words: [max(1, round(len(w) / 4)) for w in words]


def token_budget(model, default: int = 128) -> int:
    """Tokens available for text: max_length minus the [CLS] and [SEP] tokens."""
    return int(getattr(model, "max_length", default)) - 2


def chunk_text(text: str, counts: list[int], budget: int) -> list[str]:
    """Cut one long text into overlapping word-aligned chunks of <= budget tokens."""
    words = text.split()
    if sum(counts) <= budget:
        return [text]
    stride = max(1, budget // 2)
    chunks, start = [], 0
    while start < len(words):
        total, end = 0, start
        while end < len(words) and (total + counts[end] <= budget or end == start):
            total += counts[end]
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        # next chunk starts about `stride` tokens later
        moved, nxt = 0, start
        while nxt < end - 1 and moved < stride:
            moved += counts[nxt]
            nxt += 1
        start = max(nxt, start + 1)
    return chunks


def build_pieces(turns: list[Turn], count_words: Callable[[list[str]], list[int]],
                 budget: int, max_messages: int = 3) -> list[Piece]:
    """Windows of 1..max_messages consecutive turns that fit in `budget` tokens, plus
    overlapping chunks for any single turn longer than the budget."""
    sizes = []
    for t in turns:
        counts = count_words(t.text.split())
        sizes.append((sum(counts), counts))
    pieces: list[Piece] = []
    for i, t in enumerate(turns):
        n_tok, counts = sizes[i]
        if n_tok > budget:
            chunks = chunk_text(t.text, counts, budget)
            pieces += [Piece(t.index, t.index, c, 1, (k + 1, len(chunks))) for k, c in enumerate(chunks)]
            continue
        total = 0
        for j in range(i, min(len(turns), i + max_messages)):
            total += sizes[j][0]
            if total > budget:
                break
            pieces.append(Piece(t.index, turns[j].index, " ".join(x.text for x in turns[i:j + 1]), j - i + 1))
    return pieces


def score_texts(model, texts: list[str], cache: dict | None = None) -> tuple[list[float], list[str]]:
    """Malicious score (1 - P(ham)) and argmax label for each text, with optional cache."""
    import numpy as np

    cache = {} if cache is None else cache
    todo = list(dict.fromkeys(t for t in texts if t not in cache))
    if todo:
        p = np.exp(model.predict_log_proba(todo))
        ham = model.labels.index("ham")
        for t, row in zip(todo, p):
            cache[t] = (float(1 - row[ham]), model.labels[int(row.argmax())])
    return [cache[t][0] for t in texts], [cache[t][1] for t in texts]
