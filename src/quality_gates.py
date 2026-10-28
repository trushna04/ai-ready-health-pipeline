"""
quality_gates.py  (Phase 3)

Checks that decide whether the pipeline's output may move on. Two severities:
  BLOCK  failing it stops the pipeline: nothing downstream is published.
         For problems that make the data wrong or unsafe (a PHI leak, lost rows,
         two people merged into one chart).
  WARN   failing it is reported, but the run continues.
         For problems a human should look at, but that don't make the data wrong
         (more missing phones than usual, a long review queue).

Two gate stages:
  clean  runs on data/clean/ (after pipeline.py), before de-identification
  deid   runs on the STAGED de-identified tables, before they are published

Gates never read the answer key. In production there is no answer key, so every
check here has to work from the data alone.

Run (normally called by run_pipeline.py):
  python src/quality_gates.py clean
  python src/quality_gates.py deid data/staging
"""

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

import clean as cl
from data_dictionary import COLUMNS

QUALITY = Path("data/quality")

# ---------------------------------------------------------------------------
# Thresholds. Where they come from: the rates observed on seeds 7, 42 and 2026,
# plus headroom. Rates that drift past them mean the source data changed and a
# person should look before we trust the output. See BUILD_LOG for the numbers.
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "encounter_quarantine_rate": {"warn": 0.06, "block": 0.10},
    "patient_quarantine_rate":   {"warn": 0.01, "block": 0.05},
    "review_queue_share":        {"warn": 0.06},   # review pairs / clean patient records
    "max_records_per_patient":   {"warn": 4},      # 3 systems + 1 re-registration
    "row_count_drift":           {"warn": 0.25},   # vs the last run that passed every gate
    "unique_patient_share":      {"warn": 0.05, "block": 0.20},  # k-anonymity k = 1
    # Days since a source system's newest activity (visit or registration update).
    # Worst natural gap on seeds 7/42/2026 is 5 days, because these feeds are tiny
    # (about one visit every few days). A real hospital feed has hundreds of rows a
    # day, so there you'd measure this in hours, not days.
    "freshness_days":            {"warn": 7, "block": 30},
    "null_rate": {                                 # warn-only, per column
        "patient_records": {"dob": 0.05, "first_name": 0.03, "phone": 0.12, "zip": 0.12, "sex": 0.12},
        "encounters": {"diagnosis_note": 0.06, "systolic_bp": 0.18},
    },
}

# The columns downstream steps rely on. A missing one breaks them, so it blocks.
REQUIRED_COLUMNS = {
    "patient_records": ["source_record_id", "source_system", "mrn", "first_name", "last_name", "dob", "golden_patient_id"],
    "golden_patients": ["golden_patient_id", "first_name", "last_name", "dob", "sex", "zip"],
    "encounters": ["encounter_id", "source_record_id", "golden_patient_id", "admit_date", "discharge_date",
                   "length_of_stay_days", "diagnosis_note", *cl.VITAL_RANGES],
    "patients_deid": ["patient_key", "age_band", "sex", "zip3"],
    "encounters_deid": ["encounter_key", "patient_key", "admit_date_shifted", "discharge_date_shifted",
                        "diagnosis_note_scrubbed"],
}


@dataclass
class Check:
    stage: str
    name: str
    severity: str      # "block" or "warn"
    passed: bool
    detail: str

    @property
    def status(self):
        return "PASS" if self.passed else ("FAIL" if self.severity == "block" else "WARN")


def _rate_check(stage, name, rate, limits, what):
    """One metric with warn and/or block limits -> one Check, at the worst level it crosses."""
    shown = [f"{level} > {limits[level]:.0%}" for level in ("warn", "block") if level in limits]
    detail = f"{rate:.1%} {what} ({', '.join(shown)})"
    if "block" in limits and rate > limits["block"]:
        return Check(stage, name, "block", False, detail)
    if "warn" in limits and rate > limits["warn"]:
        return Check(stage, name, "warn", False, detail)
    return Check(stage, name, "block" if "block" in limits else "warn", True, detail)


# ---------------------------------------------------------------------------
# Checks shared by both stages
# ---------------------------------------------------------------------------

def check_schema(stage, tables):
    out = []
    for name, df in tables.items():
        missing = [c for c in REQUIRED_COLUMNS[name] if c not in df.columns]
        undocumented = [c for c in df.columns if c not in COLUMNS]
        problems = ([f"missing {missing}"] if missing else []) + ([f"undocumented {undocumented}"] if undocumented else [])
        out.append(Check(stage, f"schema:{name}", "block", not problems, "; ".join(problems) or "all columns present and documented"))
    return out


def check_unique(stage, df, table, key):
    dupes = int(df[key].duplicated().sum())
    nulls = int(df[key].isna().sum())
    return Check(stage, f"unique_key:{table}.{key}", "block", dupes == 0 and nulls == 0,
                 f"{dupes} duplicate, {nulls} blank")


def check_foreign_key(stage, child, child_table, parent, parent_table, key):
    orphans = int((~child[key].isin(parent[key])).sum())
    return Check(stage, f"foreign_key:{child_table}.{key}->{parent_table}", "block", orphans == 0,
                 f"{orphans} rows point to a missing {parent_table} row")


# ---------------------------------------------------------------------------
# Stage 1: clean tables
# ---------------------------------------------------------------------------

def load_clean(base=Path("data")):
    read = lambda p, **kw: pd.read_csv(base / p, dtype=str, **kw)
    return {
        "patient_records": read("clean/patient_records.csv"),
        "golden_patients": read("clean/golden_patients.csv"),
        "encounters": read("clean/encounters.csv"),
        "raw_patients": read("raw/patients.csv", keep_default_na=False),
        "raw_encounters": read("raw/encounters.csv", keep_default_na=False),
        "quarantine_patients": read("quarantine/patients.csv"),
        "quarantine_encounters": read("quarantine/encounters.csv"),
        "review_queue": read("review/match_review_queue.csv"),
    }


def check_reconciliation(stage, raw, clean_df, quarantined, table):
    """Every raw row must be accounted for: kept, quarantined, or an exact duplicate.
    Rows that vanish without a trace are the worst kind of data bug."""
    duplicates = len(raw) - len(raw.drop_duplicates())
    accounted = len(clean_df) + len(quarantined) + duplicates
    return Check(stage, f"row_reconciliation:{table}", "block", accounted == len(raw),
                 f"raw {len(raw)} = clean {len(clean_df)} + quarantined {len(quarantined)} "
                 f"+ duplicates {duplicates}" + ("" if accounted == len(raw) else f"  ({len(raw) - accounted} unaccounted)"))


def check_value_ranges(stage, records, encounters):
    """Cleaning should already have removed impossible values. If any are here, cleaning is broken."""
    problems = []
    for col, (low, high) in cl.VITAL_RANGES.items():
        v = pd.to_numeric(encounters[col], errors="coerce")
        bad = int(((v < low) | (v > high)).sum())
        if bad:
            problems.append(f"{bad} {col} outside {low}-{high}")
    los = pd.to_numeric(encounters["length_of_stay_days"], errors="coerce")
    if (los < 0).any():
        problems.append(f"{int((los < 0).sum())} negative length of stay")
    admit = pd.to_datetime(encounters["admit_date"], errors="coerce")
    if (admit > pd.Timestamp(cl.AS_OF_DATE)).any():
        problems.append(f"{int((admit > pd.Timestamp(cl.AS_OF_DATE)).sum())} admit dates after the extract date")
    dob = pd.to_datetime(records["dob"], errors="coerce")
    bad_dob = int(((dob < pd.Timestamp(cl.EARLIEST_DOB)) | (dob > pd.Timestamp(cl.AS_OF_DATE))).sum())
    if bad_dob:
        problems.append(f"{bad_dob} DOBs outside 1900-extract date")
    return Check(stage, "value_ranges", "block", not problems, "; ".join(problems) or "vitals, stays, dates all plausible")


def check_merge_conflicts(stage, records):
    """
    The wrong-merge alarm, with no answer key needed: if two records in one golden
    patient have different (formal) first names or different DOBs, the matcher has
    put two people in one chart. The matching rules should make this impossible.
    """
    g = records.groupby("golden_patient_id")
    first = g["formal_first_name"].nunique()
    dob = g["dob"].nunique()
    bad = sorted(set(first[first > 1].index) | set(dob[dob > 1].index))
    return Check(stage, "merge_conflicts", "block", not bad,
                 f"{len(bad)} golden patients mix different first names or DOBs" + (f": {bad[:5]}" if bad else ""))


def check_freshness(stage, records, encounters, as_of=cl.AS_OF_DATE):
    """
    Is every source system still sending data? A dead feed is silent: all other
    checks pass on yesterday's data, and an AI model quietly learns from a world
    that stopped updating. So we check EACH system separately; one dead feed
    hidden behind two healthy ones is the case that actually happens.

    "Newest activity" = latest admit date or registration update for that system.
    In production you'd compare with the current time; here we use the extract
    date, because the synthetic data is frozen on that day.
    """
    limits = THRESHOLDS["freshness_days"]
    newest = pd.concat([
        pd.DataFrame({"system": encounters["source_system"],
                      "when": pd.to_datetime(encounters["admit_date"], errors="coerce")}),
        pd.DataFrame({"system": records["source_system"],
                      "when": pd.to_datetime(records["record_updated_at"], errors="coerce")}),
    ]).groupby("system")["when"].max()
    age_days = {system: (pd.Timestamp(as_of) - when.normalize()).days for system, when in newest.items()}

    worst = max(age_days.values())
    severity = "block" if worst > limits["block"] else "warn"
    detail = (", ".join(f"{s} {d}d" for s, d in sorted(age_days.items()))
              + f" since newest data (warn > {limits['warn']}d, block > {limits['block']}d)")
    return Check(stage, "freshness", severity, worst <= limits["warn"], detail)


def check_null_rates(stage, tables):
    out = []
    for table, limits in THRESHOLDS["null_rate"].items():
        df = tables[table]
        for col, limit in limits.items():
            rate = df[col].isna().mean()
            out.append(_rate_check(stage, f"null_rate:{table}.{col}", rate, {"warn": limit}, "blank"))
    return out


def check_drift(stage, counts, last_good_path=QUALITY / "last_good_metrics.json"):
    """Compare row counts with the last run that passed every gate."""
    if not last_good_path.exists():
        return Check(stage, "row_count_drift", "warn", True, "no previous good run to compare with (first run)")
    previous = json.loads(last_good_path.read_text())["counts"]
    limit = THRESHOLDS["row_count_drift"]["warn"]
    moved = {k: (v - previous[k]) / previous[k] for k, v in counts.items() if previous.get(k)}
    big = {k: f"{r:+.0%}" for k, r in moved.items() if abs(r) > limit}
    return Check(stage, "row_count_drift", "warn", not big,
                 f"changed more than {limit:.0%}: {big}" if big else f"all within {limit:.0%} of last good run")


def clean_checks(t, last_good_path=QUALITY / "last_good_metrics.json"):
    s = "clean"
    records, golden, enc = t["patient_records"], t["golden_patients"], t["encounters"]
    checks = check_schema(s, {k: t[k] for k in ("patient_records", "golden_patients", "encounters")})
    checks += [
        check_unique(s, records, "patient_records", "source_record_id"),
        check_unique(s, golden, "golden_patients", "golden_patient_id"),
        check_unique(s, enc, "encounters", "encounter_id"),
        check_foreign_key(s, records, "patient_records", golden, "golden_patients", "golden_patient_id"),
        check_foreign_key(s, enc, "encounters", golden, "golden_patients", "golden_patient_id"),
        check_foreign_key(s, enc, "encounters", records, "patient_records", "source_record_id"),
        check_reconciliation(s, t["raw_patients"], records, t["quarantine_patients"], "patients"),
        check_reconciliation(s, t["raw_encounters"], enc, t["quarantine_encounters"], "encounters"),
        check_value_ranges(s, records, enc),
        check_merge_conflicts(s, records),
        check_freshness(s, records, enc),
        _rate_check(s, "quarantine_rate:encounters",
                    len(t["quarantine_encounters"]) / len(t["raw_encounters"].drop_duplicates()),
                    THRESHOLDS["encounter_quarantine_rate"], "of encounters quarantined"),
        _rate_check(s, "quarantine_rate:patients",
                    len(t["quarantine_patients"]) / len(t["raw_patients"].drop_duplicates()),
                    THRESHOLDS["patient_quarantine_rate"], "of patient rows quarantined"),
        _rate_check(s, "review_queue_size", len(t["review_queue"]) / len(records),
                    THRESHOLDS["review_queue_share"], "of records have a pair waiting for review"),
    ]
    biggest = int(records["golden_patient_id"].value_counts().max())
    checks.append(Check(s, "max_records_per_patient", "warn", biggest <= THRESHOLDS["max_records_per_patient"]["warn"],
                        f"largest golden patient has {biggest} records"))
    checks += check_null_rates(s, t)
    checks.append(check_drift(s, {"patient_records": len(records), "golden_patients": len(golden),
                                  "encounters": len(enc)}, last_good_path))
    return checks


# ---------------------------------------------------------------------------
# Stage 2: de-identified tables (staged, not yet published)
# ---------------------------------------------------------------------------

PHONE_ANYWHERE = re.compile(r"\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}\b|\b\d{10}\b")
MRN_ANYWHERE = re.compile(r"\b(?:A\d{6}|ER-\d{5}|\d{8})\b")
INTERNAL_ID = re.compile(r"\b(?:REC|GP|ENC)-\d{6}\b")


def load_deid(folder):
    return {"patients_deid": pd.read_csv(folder / "patients_deid.csv", dtype=str),
            "encounters_deid": pd.read_csv(folder / "encounters_deid.csv", dtype=str)}


def check_phi_leaks(stage, deid, records):
    """
    Scan every cell of the outgoing tables for things that must never leave:
    phone or MRN shapes, internal IDs, real names, and direct-identifier columns.
    ANY hit blocks. A single leaked phone number is a reportable breach.
    """
    names = set(records["first_name"].dropna()) | set(records["last_name"].dropna())
    full_names = {f"{f} {l}" for f, l in zip(records["first_name"], records["last_name"]) if isinstance(f, str)}
    hits = {"phone": 0, "mrn": 0, "internal_id": 0, "full_name": 0, "name_in_note": 0}
    examples = []
    for table, df in deid.items():
        for col in df.columns:
            for cell in df[col].dropna():
                found = []
                if PHONE_ANYWHERE.search(cell):
                    found.append("phone")
                if MRN_ANYWHERE.search(cell):
                    found.append("mrn")
                if INTERNAL_ID.search(cell):
                    found.append("internal_id")
                if " " in cell and any(n in cell for n in full_names):
                    found.append("full_name")
                if col == "diagnosis_note_scrubbed" and set(re.findall(r"\b[A-Z][a-z]{2,}\b", cell)) & names:
                    found.append("name_in_note")
                for f in found:
                    hits[f] += 1
                if found and len(examples) < 3:
                    examples.append(f"{table}.{col}: {cell[:60]!r}")
    direct = [f"{t}.{c}" for t, df in deid.items() for c in df.columns if COLUMNS.get(c, ("",))[0] == "direct_identifier"]
    total = sum(hits.values()) + len(direct)
    detail = (f"hits {hits}" + (f"; direct-identifier columns {direct}" if direct else "")
              + (f"; e.g. {examples}" if examples else "")) if total else "0 phones, MRNs, IDs or names in any cell"
    return Check(stage, "phi_leak_scan", "block", total == 0, detail)


def check_k_anonymity(stage, patients):
    quasi = ["age_band", "sex", "zip3"]
    k = patients.fillna("?").groupby(quasi)["patient_key"].transform("count")
    return _rate_check(stage, "k_anonymity", (k == 1).mean(), THRESHOLDS["unique_patient_share"],
                       f"of patients unique on {quasi}")


def check_secret_key(stage, strict):
    import os
    using_dev = not os.environ.get("PIPELINE_SECRET_KEY")
    return Check(stage, "secret_key", "block" if strict else "warn", not using_dev,
                 "using the development key: pseudonyms are not secure" if using_dev else "secret key set")


def check_audit_log(stage, log_path):
    from access import verify_log
    if not log_path.exists():
        return Check(stage, "audit_log_intact", "block", True, "no audit log yet")
    ok, bad = verify_log(log_path)
    return Check(stage, "audit_log_intact", "block", ok, "hash chain intact" if ok else f"chain broken at entry {bad}")


def deid_checks(deid, clean_t, strict=False, log_path=Path("data/audit/access_log.jsonl")):
    s = "deid"
    p, e = deid["patients_deid"], deid["encounters_deid"]
    checks = check_schema(s, deid)
    checks += [
        check_unique(s, p, "patients_deid", "patient_key"),
        check_unique(s, e, "encounters_deid", "encounter_key"),
        check_foreign_key(s, e, "encounters_deid", p, "patients_deid", "patient_key"),
        Check(s, "row_counts_match_clean", "block",
              len(p) == len(clean_t["golden_patients"]) and len(e) == len(clean_t["encounters"]),
              f"patients {len(p)} vs {len(clean_t['golden_patients'])}, "
              f"encounters {len(e)} vs {len(clean_t['encounters'])}"),
        check_phi_leaks(s, deid, clean_t["patient_records"]),
        check_k_anonymity(s, p),
        check_secret_key(s, strict),
        check_audit_log(s, log_path),
    ]
    return checks


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def passed(checks):
    return all(c.status != "FAIL" for c in checks)


def write_markdown_report(checks, stage, out=QUALITY):
    """
    The human-readable report: one file a data steward can open to see what
    failed, why, and what to do. The JSON file is for machines; this is for people.
    """
    verdict = "PASSED" if passed(checks) else "FAILED: pipeline stopped, nothing was published"
    lines = [f"# Quality gate report: {stage}", "",
             f"**Result: {verdict}**", "",
             "- **FAIL** blocks the pipeline. Fix the cause, then rerun.",
             "- **WARN** does not block, but a person should look before trusting the output.", "",
             "| Status | Check | Severity | Detail |", "|---|---|---|---|"]
    for c in sorted(checks, key=lambda c: ("FAIL", "WARN", "PASS").index(c.status)):
        detail = c.detail.replace("|", "/")      # a "|" would break the table
        lines.append(f"| {c.status} | `{c.name}` | {c.severity} | {detail} |")
    path = out / f"quality_report_{stage}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def report(checks, stage, out=QUALITY):
    out.mkdir(parents=True, exist_ok=True)
    (out / f"gate_report_{stage}.json").write_text(json.dumps(
        {"stage": stage, "passed": passed(checks), "checks": [asdict(c) | {"status": c.status} for c in checks]},
        indent=2))
    md_path = write_markdown_report(checks, stage, out)
    counts = {s: sum(c.status == s for c in checks) for s in ("PASS", "WARN", "FAIL")}
    print(f"\n=== Quality gate: {stage} ===  {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")
    for c in sorted(checks, key=lambda c: ("FAIL", "WARN", "PASS").index(c.status)):
        if c.status != "PASS":
            print(f"  {c.status}  {c.name:<38} {c.detail}")
    print(f"  Full report: {md_path}")
    print(f"  GATE {'PASSED' if passed(checks) else 'FAILED: pipeline stopped'}")
    return passed(checks)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "clean"
    clean_t = load_clean()
    checks = clean_checks(clean_t) if stage == "clean" else deid_checks(load_deid(Path(sys.argv[2])), clean_t)
    sys.exit(0 if report(checks, stage) else 1)


if __name__ == "__main__":
    main()
