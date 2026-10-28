"""Tests for Phase 2 governance rules. None of these need generated data. Run: pytest"""

import json

import pytest

from access import AccessDenied, read_table, verify_log, write_audit
from data_dictionary import UndocumentedColumnError, check_columns
from deidentify import age_band, date_shift_days, pseudonym, scrub_note, zip3

NAMES = {"Maria", "Garcia", "Robert", "Smith"}


# --- note scrubbing ---------------------------------------------------------

def test_phone_in_note_is_scrubbed():
    text, counts = scrub_note("LBP after lifting. Pt reachable at 610-555-0142.", NAMES)
    assert text == "LBP after lifting. Pt reachable at [PHONE]."
    assert counts == {"phone": 1}


def test_mrn_in_note_is_scrubbed():
    for mrn in ["A123456", "00412345", "ER-01234"]:
        text, _ = scrub_note(f"See prior chart MRN {mrn}.", NAMES)
        assert text == "See prior chart MRN [MRN].", mrn


def test_relative_name_scrubbed_even_if_not_in_name_list():
    text, _ = scrub_note("Spoke with pt's spouse Zelda Quinn.", NAMES)
    assert text == "Spoke with pt's spouse [NAME]."


def test_clinical_abbreviations_survive():
    note = "pt c/o CP, r/o MI. Troponin neg x2. PT referral, EKG ordered."
    assert scrub_note(note, NAMES) == (note, {})


# --- masking ----------------------------------------------------------------

def test_pseudonym_is_stable_and_depends_on_key():
    assert pseudonym(b"k1", "GP-000001", "PT") == pseudonym(b"k1", "GP-000001", "PT")
    assert pseudonym(b"k1", "GP-000001", "PT") != pseudonym(b"k2", "GP-000001", "PT")
    assert "000001" not in pseudonym(b"k1", "GP-000001", "PT")


def test_date_shift_in_range():
    shifts = {date_shift_days(b"k", f"GP-{i:06d}") for i in range(500)}
    assert min(shifts) >= 1 and max(shifts) <= 365 and len(shifts) > 100


def test_age_band_groups_90_plus():
    assert age_band(37) == "30-39"
    assert age_band(90) == age_band(104) == "90+"


def test_low_population_zip3_is_zeroed():
    assert zip3("19425") == "194"
    assert zip3("03601") == "000"


def test_undocumented_column_blocks_publishing():
    check_columns("patients_deid", ["patient_key", "sex"])
    with pytest.raises(UndocumentedColumnError):
        check_columns("patients_deid", ["patient_key", "secret_new_column"])


# --- access + audit ---------------------------------------------------------

@pytest.mark.parametrize("role,table,purpose", [
    ("analyst", "golden_patients", "reporting"),              # no PHI for analysts
    ("data_steward", "golden_patients", "just_curious"),      # PHI needs an approved purpose
    ("intern", "patients_deid", "exploring"),                 # unknown role
    ("analyst", "patients_deid", ""),                         # no purpose
])
def test_denied_requests_are_logged(tmp_path, role, table, purpose):
    log = tmp_path / "log.jsonl"
    with pytest.raises(AccessDenied):
        read_table("u", role, table, purpose=purpose, log_path=log)
    entry = json.loads(log.read_text().splitlines()[-1])
    assert entry["decision"] == "DENIED" and entry["role"] == role


def test_audit_log_detects_tampering(tmp_path):
    log = tmp_path / "log.jsonl"
    for i in range(3):
        write_audit(log, user=f"u{i}", decision="ALLOWED", rows_returned=10)
    assert verify_log(log) == (True, None)

    lines = log.read_text().splitlines()
    entry = json.loads(lines[1])
    entry["rows_returned"] = 0
    lines[1] = json.dumps(entry)
    log.write_text("\n".join(lines) + "\n")
    assert verify_log(log) == (False, 2)


def test_deleting_an_entry_is_detected(tmp_path):
    log = tmp_path / "log.jsonl"
    for i in range(3):
        write_audit(log, user=f"u{i}", decision="ALLOWED")
    lines = log.read_text().splitlines()
    log.write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert verify_log(log) == (False, 3)
