"""
data_dictionary.py  (Phase 2)

The single source of truth for what every column means and how sensitive it is.

Every column the pipeline writes must be listed here. check_columns() fails
loudly on an undocumented column, so nobody can quietly add a new PHI field
to a table that analysts can read.

Sensitivity levels:
  direct_identifier  identifies a person on its own (name, phone, MRN). One of the
                     18 HIPAA Safe Harbor identifiers.
  quasi_identifier   harmless alone, identifying in combination (DOB + ZIP + sex
                     identifies most Americans). Also a Safe Harbor identifier
                     when too precise (full dates, 5-digit ZIP).
  free_text          may contain anything, including PHI typed in by a person.
  clinical           health information; sensitive, but not an identifier.
  operational        pipeline metadata.

Run:
  python src/data_dictionary.py     (rewrites docs/data_dictionary.md)
"""

from pathlib import Path

# column -> (sensitivity, Safe Harbor identifier type or "", description)
COLUMNS = {
    # --- identifiers -----------------------------------------------------
    "first_name":        ("direct_identifier", "names", "Patient first name as registered."),
    "formal_first_name": ("direct_identifier", "names", "First name with nicknames converted (Bob -> Robert), used for matching."),
    "last_name":         ("direct_identifier", "names", "Patient last name."),
    "phone":             ("direct_identifier", "telephone numbers", "10-digit phone, digits only."),
    "mrn":               ("direct_identifier", "medical record numbers", "Medical record number, unique within one source system."),
    "patient_mrn":       ("direct_identifier", "medical record numbers", "MRN on the encounter, after leading-zero repair."),
    "source_record_id":  ("direct_identifier", "other unique identifying numbers", "ID of one registration row in the raw extract."),
    "golden_patient_id": ("direct_identifier", "other unique identifying numbers", "Pipeline's ID for one real person (all their matched records)."),
    "encounter_id":      ("direct_identifier", "other unique identifying numbers", "Source system's visit ID."),
    "dob":               ("quasi_identifier", "dates", "Date of birth."),
    "zip":               ("quasi_identifier", "geographic subdivisions", "5-digit ZIP code."),
    "admit_date":        ("quasi_identifier", "dates", "Visit start date."),
    "discharge_date":    ("quasi_identifier", "dates", "Visit end date."),
    "record_updated_at": ("quasi_identifier", "dates", "When the source system last updated the registration."),
    "sex":               ("quasi_identifier", "", "F or M."),
    # --- de-identified replacements -------------------------------------
    "patient_key":       ("operational", "", "Pseudonym for golden_patient_id: keyed hash, can't be reversed without the secret key."),
    "encounter_key":     ("operational", "", "Pseudonym for encounter_id: keyed hash."),
    "age_band":          ("quasi_identifier", "", "Age as of the extract date in 10-year bands; everyone 90+ is grouped as '90+'."),
    "age_at_admit":      ("quasi_identifier", "", "Age in whole years at the visit, capped at 90."),
    "zip3":              ("quasi_identifier", "", "First 3 ZIP digits; '000' for low-population areas (Safe Harbor rule)."),
    "admit_date_shifted":     ("quasi_identifier", "", "Admit date moved back by a secret per-patient offset (1-365 days)."),
    "discharge_date_shifted": ("quasi_identifier", "", "Discharge date moved by the same per-patient offset."),
    "diagnosis_note_scrubbed": ("free_text", "", "Diagnosis note with phones, MRNs and names replaced by [PHONE], [MRN], [NAME]."),
    # --- clinical --------------------------------------------------------
    "diagnosis_note":    ("free_text", "", "Free-text diagnosis note. Raw notes can contain leaked PHI."),
    "encounter_type":    ("clinical", "", "outpatient, emergency or inpatient."),
    "length_of_stay_days": ("clinical", "", "Discharge date minus admit date, in days."),
    "systolic_bp":       ("clinical", "", "Systolic blood pressure, mmHg. Blank if missing or implausible."),
    "diastolic_bp":      ("clinical", "", "Diastolic blood pressure, mmHg. Blank if missing or implausible."),
    "heart_rate":        ("clinical", "", "Heart rate, beats per minute. Blank if missing or implausible."),
    "insurance":         ("clinical", "", "Canonical payer name."),
    # --- operational -----------------------------------------------------
    "source_system":     ("operational", "", "clinic_a, clinic_b or er_system."),
    "source_systems":    ("operational", "", "Comma-separated systems this person appears in."),
    "n_source_records":  ("operational", "", "How many registrations were matched into this person."),
    "data_issues":       ("operational", "", "Semicolon-separated cleaning flags (e.g. dob_out_of_range)."),
}

# table -> (location, tier, description). "phi" tables: data stewards only.
TABLES = {
    "patient_records": ("data/clean/patient_records.csv", "phi", "One row per source registration, cleaned, with golden_patient_id."),
    "golden_patients": ("data/clean/golden_patients.csv", "phi", "One row per real person, best value for each field."),
    "encounters":      ("data/clean/encounters.csv", "phi", "Clean visits linked to golden_patient_id."),
    "patients_deid":   ("data/deidentified/patients_deid.csv", "deidentified", "golden_patients with identifiers removed or generalised."),
    "encounters_deid": ("data/deidentified/encounters_deid.csv", "deidentified", "encounters with pseudonyms, shifted dates and scrubbed notes."),
}


class UndocumentedColumnError(Exception):
    pass


def check_columns(table, columns):
    """Raise if a table has a column the dictionary doesn't describe."""
    missing = [c for c in columns if c not in COLUMNS]
    if missing:
        raise UndocumentedColumnError(
            f"{table} has undocumented columns {missing}. Add them to data_dictionary.COLUMNS "
            "with a sensitivity level before they can be published.")


def write_markdown(table_columns, path=Path("docs/data_dictionary.md")):
    """table_columns: {table: [column, ...]} as actually written by the pipeline."""
    lines = ["# Data dictionary", "",
             "_Generated by `python src/data_dictionary.py`. Do not edit by hand._", "",
             "Sensitivity: **direct_identifier** identifies a person alone; **quasi_identifier** "
             "identifies in combination; **free_text** may contain anything. "
             "Safe Harbor column = which of the 18 HIPAA Safe Harbor identifier types it is.", ""]
    for table, columns in table_columns.items():
        location, tier, description = TABLES[table]
        lines += [f"## `{table}`", "", f"{description}  ", f"Location: `{location}` · Tier: **{tier}**", "",
                  "| Column | Sensitivity | Safe Harbor | Description |", "|---|---|---|---|"]
        for col in columns:
            sensitivity, safe_harbor, desc = COLUMNS[col]
            lines.append(f"| `{col}` | {sensitivity} | {safe_harbor or '—'} | {desc} |")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def main():
    import pandas as pd
    table_columns = {}
    for table, (location, _, _) in TABLES.items():
        columns = list(pd.read_csv(location, nrows=0).columns)
        check_columns(table, columns)
        table_columns[table] = columns
    print(f"Wrote {write_markdown(table_columns)} ({len(table_columns)} tables)")


if __name__ == "__main__":
    main()
