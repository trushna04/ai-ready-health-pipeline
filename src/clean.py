"""
clean.py  (Phase 1)

Turns raw text values into standard, typed values. Every function here does
ONE kind of repair and is small enough to test on its own (see tests/).

Rule of thumb used everywhere: if a value can be repaired with certainty, repair
it and record a flag. If it can't, set it to None and record WHY. Never guess
silently.

This module reads nothing from disk. pipeline.py calls it.
"""

import re
from datetime import date, datetime

# The date this extract was pulled. Anything after it can't have happened yet.
# Fixed (not date.today()) so re-running next month gives the same result.
AS_OF_DATE = date(2026, 10, 26)
EARLIEST_DOB = date(1900, 1, 1)

# All the ways the source systems say "we don't know". Compared case-insensitively.
MISSING_TOKENS = {"", "n/a", "na", "null", "none", "unknown", "-"}

# Common nickname -> formal first name. Matching compares formal names,
# so "Bob Smith" and "Robert Smith" can be recognised as the same person.
# Limitation: a real system would use a much larger public nickname list.
NICKNAME_TO_FORMAL = {
    "Bob": "Robert", "Rob": "Robert", "Bill": "William", "Will": "William",
    "Mike": "Michael", "Chris": "Christopher", "Jim": "James", "Jimmy": "James",
    "Dan": "Daniel", "Danny": "Daniel", "Liz": "Elizabeth", "Beth": "Elizabeth",
    "Kate": "Katherine", "Katie": "Katherine", "Jen": "Jennifer", "Jenny": "Jennifer",
    "Peggy": "Margaret", "Maggie": "Margaret", "Pat": "Patricia", "Sue": "Susan",
    "Joe": "Joseph", "Sam": "Samuel", "Tom": "Thomas", "Tommy": "Thomas",
}

# Payer spellings seen in the source systems -> one canonical payer name.
# Keys are lowercase. Anything not listed is kept as-is and flagged, so a new
# spelling shows up in the issues report instead of being silently dropped.
INSURANCE_CANONICAL = {
    "medicare": "Medicare", "medicare part b": "Medicare",
    "medicaid": "Medicaid", "pa medicaid": "Medicaid",
    "aetna": "Aetna", "aetna ppo": "Aetna",
    "blue cross": "Blue Cross", "bcbs": "Blue Cross", "bluecross": "Blue Cross",
    "independence blue cross": "Blue Cross",
    "unitedhealthcare": "UnitedHealthcare", "uhc": "UnitedHealthcare",
    "united health": "UnitedHealthcare",
    "self-pay": "Self-pay", "self pay": "Self-pay", "self": "Self-pay",
}

SEX_CANONICAL = {"f": "F", "female": "F", "m": "M", "male": "M"}

# Plausible ranges for vitals. Outside these, the number is a data-entry error
# (999, 0, -1), not a real reading, so it's removed rather than averaged in.
VITAL_RANGES = {"systolic_bp": (50, 300), "diastolic_bp": (20, 200), "heart_rate": (20, 250)}


def blank_to_none(value):
    """'N/A', 'unknown', '-', '' ... -> None. Everything else is stripped text."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in MISSING_TOKENS else text


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    ("%Y-%m-%d", re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")),     # 1985-03-14
    ("%m/%d/%Y", re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")),     # 3/14/1985
    ("%d-%b-%Y", re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{4}$")),  # 14-Mar-1985
    ("%B %d, %Y", re.compile(r"^[A-Za-z]+ \d{1,2}, \d{4}$")),  # March 14, 1985
]
_TWO_DIGIT_YEAR = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2})$")  # 3/14/85


def parse_date(value):
    """
    Parse any of the five date formats the sources use. Returns (date or None, issue or None).

    Two-digit years are the trap: Python's strptime reads '3/4/52' as 2052.
    We use a pivot instead: a two-digit year that would land after AS_OF_DATE
    must be the 1900s. ('22' -> 2022, '52' -> 1952.)
    """
    text = blank_to_none(value)
    if text is None:
        return None, "missing"

    m = _TWO_DIGIT_YEAR.match(text)
    if m:
        month, day, yy = (int(g) for g in m.groups())
        year = 2000 + yy if 2000 + yy <= AS_OF_DATE.year else 1900 + yy
        try:
            return date(year, month, day), None
        except ValueError:
            return None, "unparseable"

    for fmt, pattern in _DATE_FORMATS:
        if pattern.match(text):
            try:
                return datetime.strptime(text, fmt).date(), None
            except ValueError:
                return None, "unparseable"
    return None, "unparseable"


def parse_dob(value):
    """parse_date plus a sanity check: born between 1900 and the extract date."""
    dob, issue = parse_date(value)
    if dob is not None and not (EARLIEST_DOB <= dob <= AS_OF_DATE):
        return None, "dob_out_of_range"
    return dob, issue


# ---------------------------------------------------------------------------
# People fields
# ---------------------------------------------------------------------------

def clean_name(value):
    """'  SMITH ' -> 'Smith'. Handles hyphens and apostrophes ('o'brien' -> "O'Brien")."""
    text = blank_to_none(value)
    if text is None:
        return None
    return re.sub(r"[A-Za-z]+", lambda w: w.group(0).capitalize(), text.lower())


def formal_first_name(first_name):
    """'Bob' -> 'Robert'. Names with no nickname entry come back unchanged."""
    if first_name is None:
        return None
    return NICKNAME_TO_FORMAL.get(first_name, first_name)


def clean_phone(value):
    """'(610) 555-0142', '+1 610.555.0142', '6105550142' -> '6105550142'."""
    text = blank_to_none(value)
    if text is None:
        return None, None
    digits = re.sub(r"\D", "", text)
    if len(digits) == 11 and digits.startswith("1"):  # leading country code
        digits = digits[1:]
    if len(digits) != 10:
        return None, "phone_invalid"
    return digits, None


def clean_zip(value):
    """'19425-1234' -> '19425'. ZIP+4 detail isn't needed for matching."""
    text = blank_to_none(value)
    if text is None:
        return None, None
    m = re.match(r"^(\d{5})(-\d{4})?$", text)
    return (m.group(1), None) if m else (None, "zip_invalid")


def clean_sex(value):
    text = blank_to_none(value)
    if text is None:
        return None, None
    sex = SEX_CANONICAL.get(text.lower())
    return (sex, None) if sex else (None, "sex_unrecognised")


def clean_insurance(value):
    text = blank_to_none(value)
    if text is None:
        return None, None
    payer = INSURANCE_CANONICAL.get(text.lower())
    return (payer, None) if payer else (text, "insurance_unmapped")


# ---------------------------------------------------------------------------
# Encounter fields
# ---------------------------------------------------------------------------

def clean_vital(name, value):
    """'120' -> 120. '999', '0', '-1' -> None with a flag. Missing -> None, no flag."""
    text = blank_to_none(value)
    if text is None:
        return None, None
    try:
        number = int(float(text))
    except ValueError:
        return None, f"{name}_not_a_number"
    low, high = VITAL_RANGES[name]
    if not (low <= number <= high):
        return None, f"{name}_implausible"
    return number, None


def repair_mrn(source_system, mrn, known_mrns):
    """
    Find the patient an encounter belongs to. Returns (mrn or None, flag or None).

    clinic_b MRNs are 8 digits with leading zeros. When a file passes through
    Excel, '00412345' becomes '412345'. Padding back to 8 digits is a SAFE
    repair only because we then confirm the padded MRN really exists.
    """
    if mrn is None:
        return None, "mrn_missing"
    if (source_system, mrn) in known_mrns:
        return mrn, None
    if source_system == "clinic_b" and mrn.isdigit() and len(mrn) < 8:
        padded = mrn.zfill(8)
        if (source_system, padded) in known_mrns:
            return padded, "mrn_leading_zeros_restored"
    return None, "orphan_mrn"
