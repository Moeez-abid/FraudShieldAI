"""Conversation parsing, token-budget windows, chunking and cross-message rules."""

from fraudshield.inference.conversation import (
    Turn,
    build_pieces,
    chunk_text,
    guess_me,
    parse_conversation,
    speakers,
)
from fraudshield.inference.conversation_rules import conversation_floor


def one_token_per_word(words):
    return [1] * len(words)


def test_parse_whatsapp_ios_and_android():
    ios = ("[24/11/2026, 10:15:32] Ali Khan: Hi, it's me\n"
           "[24/11/2026, 10:16:01] You: Who is this?\n"
           "[24/11/2026, 10:17:45] Ali Khan: My new number\nsecond line\n"
           "[24/11/2026, 10:18:00] Ali Khan: ‎image omitted")
    turns, fmt = parse_conversation(ios)
    assert fmt == "whatsapp" and [t.speaker for t in turns] == ["Ali Khan", "You", "Ali Khan"]
    assert turns[2].text == "My new number second line"
    assert guess_me(turns) == "You"

    android = "24/11/2026, 10:15 - Sara: Hey!\n24/11/2026, 10:16 - Moeez: Hi\n24/11/26, 9:01 pm - Sara: Send USDT"
    turns, fmt = parse_conversation(android)
    assert fmt == "whatsapp" and speakers(turns) == ["Sara", "Moeez"] and len(turns) == 3


def test_parse_named_lines_and_plain_lines():
    turns, fmt = parse_conversation("Unknown: hi\nYou: who?\nUnknown: new number, visit http://x.com")
    assert fmt == "named" and turns[2].speaker == "Unknown" and "http://x.com" in turns[2].text
    turns, fmt = parse_conversation("hello there\nhow are you\nsend money")
    assert fmt == "lines" and [t.speaker for t in turns] == [None] * 3
    turns, fmt = parse_conversation("first message\nstill first\n\nsecond message")
    assert fmt == "blocks" and len(turns) == 2 and turns[0].text == "first message still first"
    assert parse_conversation("   ")[0] == []


def test_windows_respect_budget_and_max_messages():
    turns = [Turn(i + 1, None, " ".join(["w"] * n)) for i, n in enumerate([3, 3, 3, 10])]
    pieces = build_pieces(turns, one_token_per_word, budget=8, max_messages=3)
    spans = {(p.start, p.end) for p in pieces}
    assert (1, 2) in spans and (1, 3) not in spans  # 9 tokens > budget 8
    assert all(p.n_messages <= 3 for p in pieces)
    # message 4 (10 tokens > 8) is cut into overlapping chunks covering every word
    chunks = [p for p in pieces if p.start == 4]
    assert len(chunks) > 1 and all(p.part for p in chunks)
    assert chunks[0].text.split()[0] == "w" and sum(len(c.text.split()) for c in chunks) >= 10


def test_chunk_text_overlaps_and_covers_all_words():
    words = [f"w{i}" for i in range(20)]
    chunks = chunk_text(" ".join(words), [1] * 20, budget=8)
    assert all(len(c.split()) <= 8 for c in chunks)
    seen = [w for c in chunks for w in c.split()]
    assert set(seen) == set(words) and len(seen) > len(words)  # overlap
    assert chunk_text("short text", [1, 1], budget=8) == ["short text"]


def turns_of(*msgs):
    return [Turn(i + 1, "them", m) for i, m in enumerate(msgs)]


def test_rule_new_number_then_money():
    floor, notes = conversation_floor(turns_of(
        "Hi beta, it's me. My phone broke, this is my new number.", "How are your studies?",
        "Can you send 20,000 on JazzCash? I'll return it tomorrow."))
    assert floor == "High" and "Message 1" in notes[0] and "message 3" in notes[0]


def test_rule_wrong_number_then_crypto():
    floor, _ = conversation_floor(turns_of("Sorry, wrong number!", "Nice to meet you anyway",
                                           "I do crypto trading, 300% returns", "Send me $200 to start"))
    assert floor == "High"
    floor, _ = conversation_floor(turns_of("Sorry, wrong number!", "I do crypto trading, it's fun"))
    assert floor == "Medium"


def test_rule_code_by_mistake_and_job_fee():
    assert conversation_floor(turns_of("I sent you a 6 digit code by mistake",
                                       "Please send me the code"))[0] == "High"
    assert conversation_floor(turns_of("Part-time job: earn Rs 5000 daily liking videos",
                                       "To start, pay the registration fee of Rs 2000"))[0] == "High"


def test_rules_quiet_on_normal_chat():
    chat = turns_of("Are we still on for Saturday?", "Let's say 7, I'll book a table.",
                    "Send me the photos from the trip when you get a chance.")
    assert conversation_floor(chat) == (None, [])
    assert conversation_floor(turns_of("Hi, it's me, new number"))[0] is None  # single message


def test_rule_investment_pitch_then_money():
    floor, notes = conversation_floor(turns_of("How was your weekend?", "I'm in a crypto trading group, I made 300% in a month.",
                                               "Send me $200 in USDT and I'll add you."))
    assert floor == "High" and "guaranteed or very high profits" in notes[0]
    assert conversation_floor(turns_of("I started trading stocks", "Can you send 5,000 for the trip?"))[0] == "Medium"
    assert conversation_floor(turns_of("I started trading stocks", "It's going ok"))[0] is None
