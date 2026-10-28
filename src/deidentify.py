"""
deidentify.py  (Phase 2)

Builds de-identified copies of the clean tables, for analysts and ML work.
The clean tables (data/clean/) still hold full PHI and stay restricted.

What happens to each kind of identifier (HIPAA Safe Harbor style):
  names, phone          removed
  MRNs, record IDs      removed; patients and visits get a PSEUDONYM: a keyed hash
                        (HMAC-SHA256). The same patient always gets the same key,
                        so tables still join, but without the secret key nobody
                        can work back to the real ID.
  date of birth         10-year age band only; 90+ is one band. Safe Harbor would allow
                        the birth year, but evaluate_phase2.py showed birth year + sex + ZIP3
                        makes over half the patients unique, so we publish less than allowed.
  ZIP                   first 3 digits; '000' for the low-population 3-digit areas
  visit dates           shifted back by a secret 1-365 day offset per patient
                        (see BUILD_LOG: this is the one place we go beyond strict Safe Harbor)
  free-text notes       phones, MRNs and names found in the text are replaced
                        with [PHONE], [MRN], [NAME]

The secret key comes from the PIPELINE_SECRET_KEY environment variable.

Run (after pipeline.py):
  python src/deidentify.py
"""

import hashlib
import hmac
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from clean import AS_OF_DATE
from data_dictionary import check_columns, write_markdown

CLEAN, OUT = Path("data/clean"), Path("data/deidentified")

DEV_KEY = "dev-only-key-not-secret"

# 3-digit ZIP areas with under 20,000 people (2000 Census), which Safe Harbor
# says must be reported as '000'. None are in this dataset; listed so the rule is real.
LOW_POPULATION_ZIP3 = {"036", "059", "063", "102", "203", "556", "692", "790",
                       "821", "823", "830", "831", "878", "879", "884", "890", "893"}


def secret_key():
    key = os.environ.get("PIPELINE_SECRET_KEY")
    if not key:
        print("WARNING: PIPELINE_SECRET_KEY not set, using the development key. "
              "Pseudonyms are NOT secure. Never do this with real data.")
        key = DEV_KEY
    return key.encode()


def pseudonym(key, value, prefix):
    """'GP-000123' -> 'PT-3f9a1c0b7e42'. Same input + same key = same output."""
    digest = hmac.new(key, f"{prefix}|{value}".encode(), hashlib.sha256).hexdigest()
    return f"{prefix}-{digest[:12]}"


def date_shift_days(key, patient_id):
    """Secret per-patient offset of 1-365 days. Every date of one patient moves by the
    same amount, so lengths of stay and gaps between visits stay true."""
    digest = hmac.new(key, f"shift|{patient_id}".encode(), hashlib.sha256).digest()
    return 1 + int.from_bytes(digest[:4], "big") % 365


def age_on(dob, on_date):
    return on_date.year - dob.year - ((on_date.month, on_date.day) < (dob.month, dob.day))


def age_band(age):
    if age is None:
        return None
    return "90+" if age >= 90 else f"{age // 10 * 10}-{age // 10 * 10 + 9}"


def zip3(zip5):
    if not zip5:
        return None
    z = str(zip5)[:3]
    return "000" if z in LOW_POPULATION_ZIP3 else z


# ---------------------------------------------------------------------------
# Free-text scrubbing
# ---------------------------------------------------------------------------

PHONE = re.compile(r"(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\b\d{3}[\s.-])\d{3}[\s.-]\d{4}\b|\b\d{10}\b")
MRN_AFTER_LABEL = re.compile(r"\b(MRN[:#]?\s*)[A-Z0-9][A-Z0-9-]*")
MRN_SHAPES = re.compile(r"\b(?:A\d{6}|ER-\d{5}|\d{8})\b")   # the three source MRN formats
RELATIVE_NAME = re.compile(r"\b(spouse|wife|husband|mother|father|son|daughter|sister|brother)"
                           r"\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?")
TITLE_CASE_WORD = re.compile(r"\b[A-Z][a-z]{2,}\b")


def scrub_note(note, known_names):
    """
    Returns (scrubbed_note, {kind: count}). Two layers for names:
      1. context: a capitalised word after 'spouse', 'mother'... is a name, known or not
      2. dictionary: any Title-case word that is a first or last name in our patient table
    ALL-CAPS words are never treated as names, so 'PT', 'MI', 'URI' survive.
    """
    if not isinstance(note, str):
        return note, {}
    counts = {}

    def sub(pattern, repl, text, kind):
        new, n = pattern.subn(repl, text)
        if n:
            counts[kind] = counts.get(kind, 0) + n
        return new

    text = sub(PHONE, "[PHONE]", note, "phone")
    text = sub(MRN_AFTER_LABEL, r"\1[MRN]", text, "mrn")
    text = sub(MRN_SHAPES, "[MRN]", text, "mrn")
    text = sub(RELATIVE_NAME, r"\1 [NAME]", text, "name")
    text = sub(TITLE_CASE_WORD, lambda m: "[NAME]" if m.group(0) in known_names else m.group(0), text, "_word")
    counts.pop("_word", None)
    if "[NAME]" in text and "name" not in counts:
        counts["name"] = text.count("[NAME]")
    text = re.sub(r"\[NAME\](\s+\[NAME\])+", "[NAME]", text)  # 'Maria [NAME]' -> one tag
    return text, counts


# ---------------------------------------------------------------------------

def main(out=OUT):
    """out: where to write. run_pipeline.py writes to a staging folder first,
    and only publishes to data/deidentified/ if the quality gates pass."""
    key = secret_key()
    out.mkdir(parents=True, exist_ok=True)
    golden = pd.read_csv(CLEAN / "golden_patients.csv", dtype=str)
    encounters = pd.read_csv(CLEAN / "encounters.csv", dtype=str)
    records = pd.read_csv(CLEAN / "patient_records.csv", dtype=str)

    known_names = set(records["first_name"].dropna()) | set(records["last_name"].dropna())
    dob_of = {g: date.fromisoformat(d) for g, d in zip(golden["golden_patient_id"], golden["dob"])
              if isinstance(d, str)}

    # --- patients ---------------------------------------------------------
    patients = []
    for g in golden.to_dict("records"):
        dob = dob_of.get(g["golden_patient_id"])
        age = age_on(dob, AS_OF_DATE) if dob else None
        patients.append({
            "patient_key": pseudonym(key, g["golden_patient_id"], "PT"),
            "age_band": age_band(age),
            "sex": g["sex"],
            "zip3": zip3(g["zip"]),
            "insurance": g["insurance"],
            "n_source_records": g["n_source_records"],
            "source_systems": g["source_systems"],
        })
    patients = pd.DataFrame(patients)

    # --- encounters -------------------------------------------------------
    rows, redactions = [], {}
    for e in encounters.to_dict("records"):
        gid = e["golden_patient_id"]
        shift = timedelta(days=date_shift_days(key, gid))
        admit, discharge = date.fromisoformat(e["admit_date"]), date.fromisoformat(e["discharge_date"])
        dob = dob_of.get(gid)
        note, counts = scrub_note(e["diagnosis_note"], known_names)
        for kind, n in counts.items():
            redactions[kind] = redactions.get(kind, 0) + n
        rows.append({
            "encounter_key": pseudonym(key, e["encounter_id"], "EN"),
            "patient_key": pseudonym(key, gid, "PT"),
            "source_system": e["source_system"],
            "encounter_type": e["encounter_type"],
            "admit_date_shifted": admit - shift,
            "discharge_date_shifted": discharge - shift,
            "length_of_stay_days": e["length_of_stay_days"],
            "age_at_admit": min(age_on(dob, admit), 90) if dob else None,
            "diagnosis_note_scrubbed": note,
            "systolic_bp": e["systolic_bp"],
            "diastolic_bp": e["diastolic_bp"],
            "heart_rate": e["heart_rate"],
            "data_issues": e["data_issues"],
        })
    encs = pd.DataFrame(rows)

    # Nothing is published unless every column is documented and classified.
    encs["age_at_admit"] = encs["age_at_admit"].astype("Int64")
    tables = {"patients_deid": patients, "encounters_deid": encs}
    for name, df in tables.items():
        check_columns(name, df.columns)
        df.to_csv(out / f"{name}.csv", index=False)

    table_columns = {"patient_records": list(records.columns), "golden_patients": list(golden.columns),
                     "encounters": list(encounters.columns),
                     **{name: list(df.columns) for name, df in tables.items()}}
    for name, cols in table_columns.items():
        check_columns(name, cols)
    write_markdown(table_columns)

    summary = {"patients_deid": len(patients), "encounters_deid": len(encs),
               "notes_with_redactions": int((encs["diagnosis_note_scrubbed"].fillna("").str.contains(r"\[(?:PHONE|MRN|NAME)\]")).sum()),
               "redactions_by_kind": redactions,
               "patients_age_90_plus": int((patients["age_band"] == "90+").sum())}
    (out / "phase2_summary.json").write_text(json.dumps(summary, indent=2))
    for k, v in summary.items():
        print(f"{k:<32} {v}")
    print(f"Wrote {out}/ and docs/data_dictionary.md")


if __name__ == "__main__":
    main()
