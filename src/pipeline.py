"""
pipeline.py  (Phase 1)

Raw CSVs in, clean tables out. Reads ONLY data/raw/, never the answer key.

Steps:
  1. Load raw files exactly as they arrived (every column as text).
  2. Drop exact duplicate rows (a batch job loaded them twice).
  3. Clean patient fields; quarantine rows that can't identify anyone.
  4. Match patients across systems -> one golden_patient_id per real person.
  5. Build one "golden record" per person from their best field values.
  6. Clean encounters, link each to its golden patient; quarantine what can't be trusted.

Outputs (all regenerated on every run):
  data/clean/patient_records.csv   one row per source registration, cleaned + golden_patient_id
  data/clean/golden_patients.csv   one row per real person (as the pipeline sees it)
  data/clean/encounters.csv        clean visits, linked to golden_patient_id
  data/quarantine/*.csv            rows held back, raw values kept, with the reason
  data/review/match_review_queue.csv  possible matches a human should decide
  data/clean/phase1_summary.json   row counts for every step

Run:
  python src/pipeline.py
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

import clean
from match_patients import match

RAW = Path("data/raw")
OUT_CLEAN, OUT_QUARANTINE, OUT_REVIEW = Path("data/clean"), Path("data/quarantine"), Path("data/review")


def load_raw(name):
    # dtype=str + keep_default_na=False: pandas must not turn "N/A" or "NULL"
    # into NaN, or "00412345" into 412345. clean.py decides what's missing.
    return pd.read_csv(RAW / f"{name}.csv", dtype=str, keep_default_na=False)


def drop_exact_duplicates(df, key, summary, name):
    """Remove rows that are identical in every column. Same key + DIFFERENT content
    is a real conflict, so those rows are returned separately for quarantine."""
    before = len(df)
    df = df.drop_duplicates()
    summary[f"{name}_exact_duplicates_dropped"] = before - len(df)
    conflicted = df[key].duplicated(keep=False)
    return df[~conflicted].copy(), df[conflicted].assign(quarantine_reason=f"conflicting_{key}")


def issues_text(issues):
    return ";".join(sorted(i for i in issues if i and i != "missing")) or None


# ---------------------------------------------------------------------------
# Patients
# ---------------------------------------------------------------------------

def clean_patient_row(row):
    issues = []
    dob, dob_issue = clean.parse_dob(row["dob"])
    issues.append(dob_issue)
    phone, issue = clean.clean_phone(row["phone"]); issues.append(issue)
    zip5, issue = clean.clean_zip(row["zip"]); issues.append(issue)
    sex, issue = clean.clean_sex(row["sex"]); issues.append(issue)
    insurance, issue = clean.clean_insurance(row["insurance"]); issues.append(issue)
    first = clean.clean_name(row["first_name"])
    return {
        "source_system": row["source_system"],
        "source_record_id": row["source_record_id"],
        "mrn": clean.blank_to_none(row["mrn"]),
        "first_name": first,
        "formal_first_name": clean.formal_first_name(first),
        "last_name": clean.clean_name(row["last_name"]),
        "dob": dob,
        "sex": sex,
        "phone": phone,
        "zip": zip5,
        "insurance": insurance,
        "record_updated_at": row["record_updated_at"],
        "data_issues": issues_text(issues),
    }


def build_golden(records):
    """
    One golden record per person. Survivorship rules:
      - name and DOB: the most common value across their records (outvotes a one-off typo);
        ties go to the most recently updated record
      - contact and insurance: the most recently updated non-missing value
    """
    rows = []
    records = records.sort_values("record_updated_at", ascending=False)
    for gid, grp in records.groupby("golden_patient_id", sort=True):
        def most_common(col):
            values = grp[col].dropna().tolist()  # already newest first, so ties go to newest
            return Counter(values).most_common(1)[0][0] if values else None

        def most_recent(col):
            values = grp[col].dropna()
            return values.iloc[0] if len(values) else None

        rows.append({
            "golden_patient_id": gid,
            "first_name": most_common("formal_first_name"),
            "last_name": most_common("last_name"),
            "dob": most_common("dob"),
            "sex": most_common("sex"),
            "phone": most_recent("phone"),
            "zip": most_recent("zip"),
            "insurance": most_recent("insurance"),
            "n_source_records": len(grp),
            "source_systems": ",".join(sorted(grp["source_system"].unique())),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Encounters
# ---------------------------------------------------------------------------

def clean_encounter_row(row, known_mrns):
    """Returns (clean_row, quarantine_reason). quarantine_reason is None if the row is usable."""
    issues = []
    mrn, mrn_flag = clean.repair_mrn(row["source_system"], clean.blank_to_none(row["patient_mrn"]), known_mrns)
    issues.append(mrn_flag)

    admit, admit_issue = clean.parse_date(row["admit_date"])
    discharge, discharge_issue = clean.parse_date(row["discharge_date"])

    vitals = {}
    for name in clean.VITAL_RANGES:
        vitals[name], issue = clean.clean_vital(name, row[name])
        issues.append(issue)

    note = clean.blank_to_none(row["diagnosis_note"])
    if note is None:
        issues.append("note_empty")

    # Quarantine: the visit can't be tied to a patient or placed in time.
    # Bad vitals and empty notes don't quarantine the visit: it still happened,
    # we just null the bad value and flag it.
    reason = None
    if mrn_flag in ("orphan_mrn", "mrn_missing"):
        reason = mrn_flag
    elif admit is None or discharge is None:
        reason = f"admit_date_{admit_issue}" if admit is None else f"discharge_date_{discharge_issue}"
    elif discharge < admit:
        reason = "discharge_before_admit"   # likely swapped, but we can't know the true dates
    elif admit > clean.AS_OF_DATE:
        reason = "admit_date_in_future"

    return {
        "encounter_id": row["encounter_id"],
        "source_system": row["source_system"],
        "patient_mrn": mrn,
        "encounter_type": clean.blank_to_none(row["encounter_type"]),
        "admit_date": admit,
        "discharge_date": discharge,
        "length_of_stay_days": (discharge - admit).days if admit and discharge else None,
        "diagnosis_note": note,
        **vitals,
        "data_issues": issues_text(issues),
    }, reason


# ---------------------------------------------------------------------------

def main():
    summary = {}
    for folder in (OUT_CLEAN, OUT_QUARANTINE, OUT_REVIEW):
        folder.mkdir(parents=True, exist_ok=True)

    # --- Patients ---------------------------------------------------------
    raw_patients = load_raw("patients")
    summary["raw_patient_rows"] = len(raw_patients)
    patients, patient_conflicts = drop_exact_duplicates(raw_patients, "source_record_id", summary, "patient")

    cleaned = pd.DataFrame([clean_patient_row(r) for r in patients.to_dict("records")])
    # A record we can't identify (no MRN or no last name) can't be matched or linked.
    unusable = cleaned["mrn"].isna() | cleaned["last_name"].isna()
    patient_quarantine = pd.concat([
        patient_conflicts,
        patients[unusable.values].assign(quarantine_reason="missing_mrn_or_last_name"),
    ])
    cleaned = cleaned[~unusable].reset_index(drop=True)

    # --- Matching ---------------------------------------------------------
    # pandas stores missing values as NaN, and NaN is not None: without this,
    # a missing sex would look like a DIFFERENT sex to the matcher and veto the pair.
    records = cleaned.astype(object).where(cleaned.notna(), None).to_dict("records")
    golden_ids, pair_results = match(records)
    cleaned["golden_patient_id"] = golden_ids
    golden = build_golden(cleaned)

    decisions = Counter(p["decision"] for p in pair_results)
    summary["candidate_pairs_compared"] = len(pair_results)
    summary["pairs_auto_matched"] = decisions["auto_match"]
    summary["pairs_sent_to_review"] = decisions["review"]
    summary["clean_patient_records"] = len(cleaned)
    summary["golden_patients"] = len(golden)

    # Review queue: skip pairs that ended up in the same person anyway via other links.
    review = []
    for p in pair_results:
        a, b = records[p["i"]], records[p["j"]]
        if p["decision"] != "review" or golden_ids[p["i"]] == golden_ids[p["j"]]:
            continue
        review.append({
            "record_a": a["source_record_id"], "record_b": b["source_record_id"],
            "golden_a": golden_ids[p["i"]], "golden_b": golden_ids[p["j"]],
            "score": p["score"], "reasons": p["reasons"],
            **{f"{f}_a": a[f] for f in ("first_name", "last_name", "dob", "phone", "zip")},
            **{f"{f}_b": b[f] for f in ("first_name", "last_name", "dob", "phone", "zip")},
        })
    review = pd.DataFrame(review).sort_values("score", ascending=False)
    summary["review_queue_pairs"] = len(review)

    # --- Encounters -------------------------------------------------------
    raw_encounters = load_raw("encounters")
    summary["raw_encounter_rows"] = len(raw_encounters)
    encounters, encounter_conflicts = drop_exact_duplicates(raw_encounters, "encounter_id", summary, "encounter")

    mrn_to_record = {(r["source_system"], r["mrn"]): r for r in cleaned.to_dict("records")}
    good, held = [], [encounter_conflicts]
    for raw_row in encounters.to_dict("records"):
        row, reason = clean_encounter_row(raw_row, mrn_to_record)
        if reason:
            held.append(pd.DataFrame([{**raw_row, "quarantine_reason": reason}]))
            continue
        patient = mrn_to_record[(row["source_system"], row["patient_mrn"])]
        row["source_record_id"] = patient["source_record_id"]
        row["golden_patient_id"] = patient["golden_patient_id"]
        good.append(row)
    clean_encounters = pd.DataFrame(good)
    encounter_quarantine = pd.concat(held, ignore_index=True)
    summary["clean_encounters"] = len(clean_encounters)
    summary["quarantined_encounters"] = len(encounter_quarantine)
    summary["quarantined_patients"] = len(patient_quarantine)
    summary["encounter_quarantine_reasons"] = encounter_quarantine["quarantine_reason"].value_counts().to_dict()
    summary["patient_field_issues"] = (cleaned["data_issues"].dropna().str.split(";").explode()
                                       .value_counts().to_dict())
    summary["encounter_field_issues"] = (clean_encounters["data_issues"].dropna().str.split(";").explode()
                                         .value_counts().to_dict())

    # --- Write ------------------------------------------------------------
    cleaned.to_csv(OUT_CLEAN / "patient_records.csv", index=False)
    golden.to_csv(OUT_CLEAN / "golden_patients.csv", index=False)
    clean_encounters.to_csv(OUT_CLEAN / "encounters.csv", index=False)
    patient_quarantine.to_csv(OUT_QUARANTINE / "patients.csv", index=False)
    encounter_quarantine.to_csv(OUT_QUARANTINE / "encounters.csv", index=False)
    review.to_csv(OUT_REVIEW / "match_review_queue.csv", index=False)
    (OUT_CLEAN / "phase1_summary.json").write_text(json.dumps(summary, indent=2))

    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"    {kk:<34} {vv}")
        else:
            print(f"{k:<38} {v}")


if __name__ == "__main__":
    main()
