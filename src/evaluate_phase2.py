"""
evaluate_phase2.py  (Phase 2)

Checks that de-identification actually worked, instead of trusting it.

  1. Note scrubbing: of the notes that really leaked PHI, how many were caught,
     and were any clean notes damaged? (Grading uses the generator's note
     templates as ground truth: a note that isn't a pure template had PHI added.)
  2. Leak scan: search EVERY cell of the de-identified tables for any real
     phone, MRN, record ID, name pair, DOB or 5-digit ZIP from the PHI tables.
  3. Re-identification risk: k-anonymity on (birth_year, sex, zip3). k=1 means
     a patient is the only one with that combination, so they are easier to single out.
  4. Date shifting kept lengths of stay and visit gaps true.
  5. The audit log chain is intact, and a tampered copy is detected.

Run (after deidentify.py and access.py):
  python src/evaluate_phase2.py
"""

import json
import re
import shutil
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from access import AUDIT_LOG, verify_log
from deidentify import pseudonym, secret_key
from generate_messy_data import NOTE_TEMPLATES

CLEAN, DEID = Path("data/clean"), Path("data/deidentified")


def main():
    records = pd.read_csv(CLEAN / "patient_records.csv", dtype=str)
    golden = pd.read_csv(CLEAN / "golden_patients.csv", dtype=str)
    enc = pd.read_csv(CLEAN / "encounters.csv", dtype=str)
    pdeid = pd.read_csv(DEID / "patients_deid.csv", dtype=str)
    edeid = pd.read_csv(DEID / "encounters_deid.csv", dtype=str)
    results = {}

    # --- 1. Note scrubbing ------------------------------------------------
    templates = {t for ts in NOTE_TEMPLATES.values() for t in ts}
    notes = pd.DataFrame({"raw": enc["diagnosis_note"], "scrubbed": edeid["diagnosis_note_scrubbed"]}).dropna()
    leaked = notes[~notes["raw"].isin(templates)]
    clean_notes = notes[notes["raw"].isin(templates)]
    names = set(records["first_name"].dropna()) | set(records["last_name"].dropna())

    def residual_phi(text):
        words = set(re.findall(r"\b[A-Z][a-z]{2,}\b", text))
        return bool(re.search(r"\d{3}-\d{3}-\d{4}|\d{10}|MRN [A-Z0-9]", text) or words & names)

    results["note_scrubbing"] = {
        "notes_with_leaked_phi": len(leaked),
        "caught": int((leaked["raw"] != leaked["scrubbed"]).sum()),
        "still_contain_phi_after_scrub": int(leaked["scrubbed"].map(residual_phi).sum()),
        "clean_notes_wrongly_changed": int((clean_notes["raw"] != clean_notes["scrubbed"]).sum()),
        "clean_notes_checked": len(clean_notes),
    }

    # --- 2. Leak scan -----------------------------------------------------
    secrets = {
        "phone": set(records["phone"].dropna()),
        "mrn": set(records["mrn"].dropna()) | set(enc["patient_mrn"].dropna()),
        "record_or_patient_id": set(records["source_record_id"]) | set(golden["golden_patient_id"]) | set(enc["encounter_id"]),
        "zip5": set(records["zip"].dropna()),
    }
    full_names = {f"{f} {l}" for f, l in zip(records["first_name"], records["last_name"]) if isinstance(f, str)}
    hits = {k: 0 for k in [*secrets, "full_name"]}
    for table in (pdeid, edeid):
        for cell in table.fillna("").to_numpy().ravel():
            digits = re.sub(r"\D", "", cell)
            tokens = set(re.findall(r"[A-Za-z0-9-]+", cell))
            hits["phone"] += any(p in digits for p in secrets["phone"]) if len(digits) >= 10 else 0
            hits["mrn"] += bool(tokens & secrets["mrn"])
            hits["record_or_patient_id"] += bool(tokens & secrets["record_or_patient_id"])
            hits["zip5"] += bool(set(re.findall(r"\b\d{5}\b", cell)) & secrets["zip5"])
            hits["full_name"] += any(n in cell for n in full_names) if " " in cell else 0
    # A shifted visit date can land on SOME child's birthday by coincidence. What matters
    # is whether a patient's OWN DOB appears in their OWN rows, so check per patient.
    key = secret_key()
    key_of = {g: pseudonym(key, g, "PT") for g in golden["golden_patient_id"]}
    own_dob = {key_of[g]: d for g, d in zip(golden["golden_patient_id"], golden["dob"]) if isinstance(d, str)}
    hits["own_dob_in_own_rows"] = int(sum(
        own_dob.get(row[0]) in row[1:] for t in (pdeid, edeid)
        for row in t[["patient_key", *[c for c in t.columns if c != "patient_key"]]].fillna("").to_numpy().tolist()))
    results["leak_scan_hits"] = hits

    # --- 3. k-anonymity ---------------------------------------------------
    # Compare what we publish (age_band) with what Safe Harbor would ALLOW (birth year).
    def k_report(df, quasi):
        k = df.fillna("?").groupby(quasi)[quasi[0]].transform("count")
        return {"quasi_identifiers": quasi, "min_k": int(k.min()), "unique_k1": int((k == 1).sum()),
                "k_below_5": int((k < 5).sum()), "patients": len(df)}

    safe_harbor_allowed = pd.DataFrame({"birth_year": golden["dob"].str[:4], "sex": golden["sex"],
                                        "zip3": golden["zip"].str[:3]})
    results["k_anonymity_if_birth_year_published"] = k_report(safe_harbor_allowed, ["birth_year", "sex", "zip3"])
    results["k_anonymity_published"] = k_report(pdeid, ["age_band", "sex", "zip3"])

    # --- 4. Date shifting -------------------------------------------------
    real = enc.assign(a=pd.to_datetime(enc["admit_date"]), d=pd.to_datetime(enc["discharge_date"]))
    shifted = edeid.assign(a=pd.to_datetime(edeid["admit_date_shifted"]), d=pd.to_datetime(edeid["discharge_date_shifted"]))
    gaps_real = real.sort_values("a").groupby("golden_patient_id")["a"].apply(lambda s: list(s.diff().dropna()))
    gaps_shift = shifted.sort_values("a").groupby("patient_key")["a"].apply(lambda s: list(s.diff().dropna()))
    results["date_shift"] = {
        "length_of_stay_preserved": bool(((real["d"] - real["a"]) == (shifted["d"] - shifted["a"])).all()),
        "visit_gaps_preserved": sorted(map(str, gaps_real)) == sorted(map(str, gaps_shift)),
        "any_real_date_unchanged": bool((real["a"] == shifted["a"]).any()),
    }

    # --- 5. Audit log -----------------------------------------------------
    ok, _ = verify_log(AUDIT_LOG)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "log.jsonl"
        shutil.copy(AUDIT_LOG, copy)
        lines = copy.read_text().splitlines()
        first = json.loads(lines[0])
        first["rows_returned"] = 0          # someone hides how much data they took
        lines[0] = json.dumps(first)
        copy.write_text("\n".join(lines) + "\n")
        tampered_ok, bad_seq = verify_log(copy)
    results["audit_log"] = {"entries": len(AUDIT_LOG.read_text().splitlines()), "chain_intact": ok,
                            "tampered_copy_detected": not tampered_ok, "tamper_found_at_seq": bad_seq}

    Path("data/eval").mkdir(exist_ok=True)
    Path("data/eval/phase2_metrics.json").write_text(json.dumps(results, indent=2))
    for section, values in results.items():
        print(f"\n=== {section} ===")
        for k_, v in values.items():
            print(f"  {k_:<36} {v}")


if __name__ == "__main__":
    main()
