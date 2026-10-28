"""
demo_gate_failures.py  (Phase 3)

A gate that has never failed hasn't been tested yet. This script breaks a copy
of the real outputs in nine different ways and confirms the right gate blocks
each one. Nothing on disk is changed: every fault is applied to in-memory copies
(the audit-log fault uses a temporary copy of the log).

Run (after run_pipeline.py):
  python src/demo_gate_failures.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

import quality_gates as gates


def copy_tables(t):
    return {k: v.copy() for k, v in t.items()}


def fault_lost_rows(c, d):
    c["encounters"] = c["encounters"].iloc[5:]                      # 5 visits vanish silently
    return "clean", "row_reconciliation:encounters"


def fault_wrong_merge(c, d):
    r = c["patient_records"]
    other = r.loc[r["formal_first_name"] != r.loc[0, "formal_first_name"], "golden_patient_id"].iloc[0]
    r.loc[0, "golden_patient_id"] = other                           # two different people, one chart
    return "clean", "merge_conflicts"


def fault_orphan_encounter(c, d):
    c["encounters"].loc[c["encounters"].index[0], "golden_patient_id"] = "GP-999999"
    return "clean", "foreign_key:encounters.golden_patient_id->golden_patients"


def fault_impossible_vitals(c, d):
    c["encounters"].loc[c["encounters"].index[:3], "systolic_bp"] = "999"   # cleaning step "broke"
    return "clean", "value_ranges"


def fault_quarantine_spike(c, d):
    enc = c["encounters"]
    moved = enc.sample(frac=0.12, random_state=1)                   # an upstream system starts sending junk
    c["encounters"] = enc.drop(moved.index)
    c["quarantine_encounters"] = pd.concat([c["quarantine_encounters"], moved.assign(quarantine_reason="orphan_mrn")])
    return "clean", "quarantine_rate:encounters"


def fault_stale_feed(c, d):
    """The ER interface silently stopped 60 days ago: its newest data is 2 months old."""
    days = pd.Timedelta(days=60)
    enc, rec = c["encounters"], c["patient_records"]
    er = enc["source_system"] == "er_system"
    enc.loc[er, "admit_date"] = (pd.to_datetime(enc.loc[er, "admit_date"]) - days).dt.strftime("%Y-%m-%d")
    er = rec["source_system"] == "er_system"
    rec.loc[er, "record_updated_at"] = (pd.to_datetime(rec.loc[er, "record_updated_at"]) - days).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return "clean", "freshness"


def fault_phi_in_note(c, d):
    notes = d["encounters_deid"]["diagnosis_note_scrubbed"]
    i = notes.first_valid_index()
    d["encounters_deid"].loc[i, "diagnosis_note_scrubbed"] = notes[i] + " Call back at 610-555-0142."  # scrubber regression
    return "deid", "phi_leak_scan"


def fault_undocumented_column(c, d):
    d["patients_deid"]["ssn"] = "000-00-0000"                       # someone adds a column without classifying it
    return "deid", "schema:patients_deid"


def fault_tampered_audit_log(c, d, log_path):
    lines = log_path.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["rows_returned"] = 0                                      # hide how much data was taken
    lines[0] = json.dumps(entry)
    log_path.write_text("\n".join(lines) + "\n")
    return "deid", "audit_log_intact"


FAULTS = [fault_lost_rows, fault_wrong_merge, fault_orphan_encounter, fault_impossible_vitals,
          fault_quarantine_spike, fault_stale_feed, fault_phi_in_note, fault_undocumented_column, fault_tampered_audit_log]


def main():
    clean_t = gates.load_clean()
    deid_t = gates.load_deid(Path("data/deidentified"))
    real_log = Path("data/audit/access_log.jsonl")
    caught = 0
    print(f"{'fault':<28}{'gate':<7}{'expected check':<58}result")
    with tempfile.TemporaryDirectory() as tmp:
        for fault in FAULTS:
            c, d = copy_tables(clean_t), copy_tables(deid_t)
            log_copy = Path(tmp) / "log.jsonl"
            if real_log.exists():
                shutil.copy(real_log, log_copy)
            else:
                from access import write_audit
                log_copy.unlink(missing_ok=True)
                write_audit(log_copy, user="demo", decision="ALLOWED", rows_returned=10)
            if fault is fault_tampered_audit_log:
                stage, expected = fault(c, d, log_copy)
            else:
                stage, expected = fault(c, d)
            checks = (gates.clean_checks(c, Path(tmp) / "none.json") if stage == "clean"
                      else gates.deid_checks(d, c, log_path=log_copy))
            failed = {ch.name for ch in checks if ch.status == "FAIL"}
            ok = expected in failed and not gates.passed(checks)
            caught += ok
            print(f"{fault.__name__.removeprefix('fault_'):<28}{stage:<7}{expected:<58}"
                  f"{'BLOCKED' if ok else 'NOT CAUGHT'}")
    print(f"\n{caught} / {len(FAULTS)} injected faults blocked")
    sys.exit(0 if caught == len(FAULTS) else 1)


if __name__ == "__main__":
    main()
