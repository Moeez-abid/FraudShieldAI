"""Context rules: raise-only floors from optional user answers."""

import pytest

from fraudshield.inference.context_rules import (
    NO,
    SENDER_CONTACT,
    SENDER_OFFICIAL,
    SENDER_UNKNOWN,
    YES,
    Context,
    context_floor,
)

NEW_NUMBER = "Hi Ammi, it's me. My phone broke so this is my new number. Save it and message me here."
NEW_NUMBER_MONEY = "Hi mum it's me, new number. Please send Rs 15,000 on JazzCash urgently, I'll explain later."
WHATSAPP_CODE = "Hi, I sent you a 6 digit code by mistake, please send me the code."
REAL_OTP = "Your HBL OTP is 482913. Do not share this code with anyone. HBL will never ask for it."
BANK_ALERT = "HBL: Rs 2,500 debited from your account on 24-Nov via Debit Card. Avl bal Rs 31,200."
PARCEL = "Your parcel is held at the depot. Pay Rs 150 redelivery fee at the link to release it."


def test_no_context_changes_nothing():
    for text in (NEW_NUMBER, WHATSAPP_CODE, BANK_ALERT, PARCEL):
        assert context_floor(text, None) == (None, [])
        assert context_floor(text, Context()) == (None, [])


def test_invalid_values_rejected():
    with pytest.raises(ValueError):
        Context(sender="stranger")
    with pytest.raises(ValueError):
        Context(expected="maybe")


def test_family_on_new_number():
    assert context_floor(NEW_NUMBER, Context(sender=SENDER_UNKNOWN))[0] == "Medium"
    assert context_floor(NEW_NUMBER_MONEY, Context(sender=SENDER_UNKNOWN))[0] == "High"
    # From a saved contact this rule doesn't apply
    assert context_floor(NEW_NUMBER, Context(sender=SENDER_CONTACT))[0] is None


def test_code_request_is_high_but_real_otp_is_not():
    assert context_floor(WHATSAPP_CODE, Context(sender=SENDER_CONTACT))[0] == "High"
    assert context_floor(REAL_OTP, Context(sender=SENDER_OFFICIAL, expected=YES))[0] is None


def test_organisation_from_personal_number():
    floor, notes = context_floor(BANK_ALERT, Context(sender=SENDER_UNKNOWN))
    assert floor == "High" and "ordinary mobile number" in notes[0]
    assert context_floor(BANK_ALERT, Context(sender=SENDER_OFFICIAL))[0] is None


def test_unexpected_transactional():
    assert context_floor(PARCEL, Context(expected=NO), has_urls=True)[0] == "High"
    assert context_floor("Your order has been shipped.", Context(expected=NO))[0] == "Medium"
    assert context_floor(PARCEL, Context(expected=YES), has_urls=True)[0] is None


def test_no_account_with_named_organisation():
    threat = "Your Meezan account is blocked. Verify your account to restore access."
    assert context_floor(threat, Context(has_account=NO))[0] == "High"
    assert context_floor("Meezan Bank is open on Saturday.", Context(has_account=NO))[0] == "Medium"
    assert context_floor(threat, Context(has_account=YES))[0] is None


def test_reassuring_context_never_lowers():
    # The most reassuring answers produce no floor (so never a lower rating)
    calm = Context(sender=SENDER_CONTACT, expected=YES, has_account=YES)
    for text in (NEW_NUMBER, BANK_ALERT, PARCEL):
        assert context_floor(text, calm, has_urls=True)[0] is None
