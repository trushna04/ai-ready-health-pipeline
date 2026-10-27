"""Small tests for the Phase 1 rules most likely to break silently. Run: pytest"""

from datetime import date

import clean
from match_patients import compare, edit_distance_at_most_one


# --- cleaning ---------------------------------------------------------------

def test_all_missing_spellings_become_none():
    for token in ["", "N/A", "NULL", "unknown", "none", "-", "  n/a "]:
        assert clean.blank_to_none(token) is None


def test_every_date_format_parses_to_the_same_day():
    for text in ["1985-03-14", "3/14/1985", "3/14/85", "14-Mar-1985", "March 14, 1985"]:
        assert clean.parse_date(text) == (date(1985, 3, 14), None), text


def test_two_digit_year_pivot():
    # Python's strptime would read '52' as 2052.
    assert clean.parse_date("3/4/52")[0] == date(1952, 3, 4)
    assert clean.parse_date("10/30/22")[0] == date(2022, 10, 30)


def test_impossible_dob_is_removed_not_kept():
    assert clean.parse_dob("2085-01-01") == (None, "dob_out_of_range")
    assert clean.parse_dob("07-Apr-1869") == (None, "dob_out_of_range")


def test_phone_formats():
    for text in ["(610) 555-0142", "610-555-0142", "6105550142", "+1 610.555.0142"]:
        assert clean.clean_phone(text) == ("6105550142", None)


def test_vitals():
    assert clean.clean_vital("systolic_bp", "120") == (120, None)
    assert clean.clean_vital("systolic_bp", "999") == (None, "systolic_bp_implausible")
    assert clean.clean_vital("systolic_bp", "N/A") == (None, None)


def test_mrn_zero_padding_only_when_padded_mrn_exists():
    known = {("clinic_b", "00412345")}
    assert clean.repair_mrn("clinic_b", "412345", known) == ("00412345", "mrn_leading_zeros_restored")
    assert clean.repair_mrn("clinic_b", "999999", known) == (None, "orphan_mrn")


# --- matching ---------------------------------------------------------------

def rec(**overrides):
    base = {"source_record_id": "REC-1", "first_name": "Robert", "formal_first_name": "Robert",
            "last_name": "Smith", "dob": date(1980, 5, 1), "sex": "M",
            "phone": "6105550142", "zip": "19425"}
    return {**base, **overrides}


def test_typo_distance():
    assert edit_distance_at_most_one("Smith", "Smtih")
    assert edit_distance_at_most_one("Smith", "Smyth")
    assert not edit_distance_at_most_one("Smith", "Smithers")


def test_nickname_and_typo_still_match():
    decision, _, _ = compare(rec(), rec(first_name="Bob", last_name="Smtih"))
    assert decision == "auto_match"


def test_twins_never_merge():
    # Same last name, DOB, phone, ZIP. Different first name = different person.
    decision, _, _ = compare(rec(), rec(first_name="William", formal_first_name="William"))
    assert decision == "non_match"


def test_missing_first_name_goes_to_review_not_auto():
    # Could be Robert, could be his twin: a human must decide.
    decision, _, _ = compare(rec(), rec(first_name=None, formal_first_name=None))
    assert decision == "review"


def test_missing_dob_goes_to_review_not_auto():
    decision, _, _ = compare(rec(), rec(dob=None))
    assert decision == "review"


def test_household_member_does_not_match():
    decision, _, _ = compare(rec(), rec(dob=date(1952, 9, 9)))
    assert decision == "non_match"
