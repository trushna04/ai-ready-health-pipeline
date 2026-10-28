"""
access.py  (Phase 2)

Role-based access to the tables, with every request written to an audit log.

Rules, from HIPAA's "minimum necessary" idea: each role sees only what its job needs.
  data_steward  full PHI tables, but only for an approved purpose (e.g. reviewing matches)
  ml_engineer   de-identified tables, including scrubbed notes (needed for labelling)
  analyst       de-identified tables, WITHOUT notes (counts and trends don't need free text)
Anything not granted is denied, and the denial is logged too.

The audit log is append-only JSON lines. Each entry stores the hash of the entry
before it, so editing or deleting any old entry breaks the chain, and verify_log()
reports exactly where. (Tamper-EVIDENT, not tamper-proof: someone with file access can
still rewrite everything, which is why production logs go to a separate locked store.)

Run (after deidentify.py):
  python src/access.py     (runs a short demo of allowed and denied requests)
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_dictionary import TABLES

AUDIT_LOG = Path("data/audit/access_log.jsonl")

ALL = "*"
POLICY = {
    "data_steward": {table: ALL for table in TABLES},
    "ml_engineer": {"patients_deid": ALL, "encounters_deid": ALL},
    "analyst": {"patients_deid": ALL,
                "encounters_deid": {"encounter_key", "patient_key", "source_system", "encounter_type",
                                    "admit_date_shifted", "discharge_date_shifted", "length_of_stay_days",
                                    "age_at_admit", "systolic_bp", "diastolic_bp", "heart_rate", "data_issues"}},
}

# Reading full-PHI tables needs one of these stated reasons.
APPROVED_PHI_PURPOSES = {"patient_matching_review", "data_quality_investigation"}


class AccessDenied(Exception):
    pass


def _hash(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()


def _last_entry(log_path):
    if not log_path.exists():
        return None
    lines = log_path.read_text().splitlines()
    return json.loads(lines[-1]) if lines else None


def write_audit(log_path=AUDIT_LOG, **fields):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last = _last_entry(log_path)
    entry = {"seq": (last["seq"] + 1) if last else 1,
             "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             **fields,
             "prev_hash": last["entry_hash"] if last else "GENESIS"}
    entry["entry_hash"] = _hash(entry)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def verify_log(log_path=AUDIT_LOG):
    """Returns (True, None) if the chain is intact, else (False, seq of the first bad entry)."""
    prev = "GENESIS"
    for line in log_path.read_text().splitlines():
        entry = json.loads(line)
        stored = entry.pop("entry_hash")
        if entry["prev_hash"] != prev or _hash(entry) != stored:
            return False, entry.get("seq")
        prev = stored
    return True, None


def read_table(user, role, table, columns=None, purpose="", log_path=AUDIT_LOG):
    """
    The only way to read a table. Returns a DataFrame, or raises AccessDenied.
    columns=None means 'every column my role is allowed to see'.
    """
    def deny(reason):
        write_audit(log_path, user=user, role=role, table=table, purpose=purpose,
                    columns_requested=columns, decision="DENIED", reason=reason, rows_returned=0)
        raise AccessDenied(reason)

    if not purpose:
        deny("no purpose given")
    grants = POLICY.get(role)
    if grants is None:
        deny(f"unknown role '{role}'")
    if table not in grants:
        deny(f"role '{role}' has no access to table '{table}'")
    if TABLES[table][1] == "phi" and purpose not in APPROVED_PHI_PURPOSES:
        deny(f"purpose '{purpose}' is not approved for PHI tables")

    location = TABLES[table][0]
    available = list(pd.read_csv(location, nrows=0).columns)
    allowed = available if grants[table] == ALL else [c for c in available if c in grants[table]]
    if columns is not None:
        blocked = [c for c in columns if c not in allowed]
        if blocked:
            deny(f"role '{role}' may not read columns {blocked}")
        allowed = [c for c in allowed if c in columns]

    df = pd.read_csv(location, dtype=str, usecols=allowed)[allowed]
    write_audit(log_path, user=user, role=role, table=table, purpose=purpose,
                columns_requested=columns, columns_returned=allowed, decision="ALLOWED",
                reason=None, rows_returned=len(df))
    return df


def main():
    requests = [
        ("ana",   "analyst",      "encounters_deid", None, "monthly_visit_volume"),
        ("ana",   "analyst",      "encounters_deid", ["diagnosis_note_scrubbed"], "monthly_visit_volume"),
        ("ana",   "analyst",      "golden_patients", None, "monthly_visit_volume"),
        ("maya",  "ml_engineer",  "encounters_deid", ["encounter_key", "diagnosis_note_scrubbed"], "diagnosis_labelling"),
        ("sam",   "data_steward", "patient_records", None, "patient_matching_review"),
        ("sam",   "data_steward", "golden_patients", None, "just_curious"),
        ("guest", "intern",       "patients_deid",   None, "exploring"),
    ]
    for user, role, table, columns, purpose in requests:
        try:
            df = read_table(user, role, table, columns, purpose)
            print(f"ALLOWED  {user:<6}{role:<14}{table:<17} {len(df)} rows x {len(df.columns)} cols")
        except AccessDenied as e:
            print(f"DENIED   {user:<6}{role:<14}{table:<17} {e}")

    ok, bad = verify_log()
    n = len(AUDIT_LOG.read_text().splitlines())
    print(f"\nAudit log: {AUDIT_LOG}  ({n} entries)  chain intact: {ok}" + ("" if ok else f", first bad entry: {bad}"))


if __name__ == "__main__":
    main()
