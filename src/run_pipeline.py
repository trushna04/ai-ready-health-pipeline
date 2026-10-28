"""
run_pipeline.py  (Phase 3)

Runs every step in order, with quality gates between them. If a gate FAILS,
the run stops and nothing new is published. Analysts and ML jobs keep reading
the last good version in data/deidentified/, never a half-checked one.

  1. pipeline.py      raw -> clean, matched, quarantined
  2. GATE: clean      schema, keys, row reconciliation, ranges, wrong-merge alarm, ...
  3. deidentify.py    clean -> data/staging/   (not visible to anyone yet)
  4. GATE: deid       PHI leak scan, k-anonymity, keys, row counts, audit log
  5. publish          data/staging/ -> data/deidentified/  (swapped in all at once)

Every run is appended to data/quality/run_history.jsonl, pass or fail.

Run:
  python src/run_pipeline.py            (dev: a missing secret key is a warning)
  python src/run_pipeline.py --strict   (prod: a missing secret key blocks the run)
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import deidentify
import pipeline
import quality_gates as gates

STAGING, PUBLISHED = Path("data/staging"), Path("data/deidentified")
HISTORY = gates.QUALITY / "run_history.jsonl"
LAST_GOOD = gates.QUALITY / "last_good_metrics.json"


def publish(staging, published):
    """Swap the whole folder in one step, so readers never see a mix of old and new files."""
    incoming = published.with_name(published.name + ".incoming")
    shutil.rmtree(incoming, ignore_errors=True)
    shutil.copytree(staging, incoming)
    shutil.rmtree(published, ignore_errors=True)
    incoming.rename(published)


def record_run(outcome, stopped_at, checks, started):
    gates.QUALITY.mkdir(parents=True, exist_ok=True)
    entry = {"started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "outcome": outcome, "stopped_at": stopped_at,
             "fail": [c.name for c in checks if c.status == "FAIL"],
             "warn": [c.name for c in checks if c.status == "WARN"]}
    with open(HISTORY, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run the pipeline with quality gates.")
    parser.add_argument("--strict", action="store_true", help="production mode: dev secret key blocks the run")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print("--- Step 1: clean, match, quarantine ---")
    pipeline.main()

    clean_t = gates.load_clean()
    clean_checks = gates.clean_checks(clean_t, LAST_GOOD)
    if not gates.report(clean_checks, "clean"):
        record_run("FAILED", "gate:clean", clean_checks, started)
        print("\nStopped before de-identification. Published data was NOT changed.")
        sys.exit(1)

    print("\n--- Step 3: de-identify into staging ---")
    shutil.rmtree(STAGING, ignore_errors=True)
    deidentify.main(out=STAGING)

    deid_checks = gates.deid_checks(gates.load_deid(STAGING), clean_t, strict=args.strict)
    all_checks = clean_checks + deid_checks
    if not gates.report(deid_checks, "deid"):
        record_run("FAILED", "gate:deid", all_checks, started)
        print(f"\nStopped before publishing. {PUBLISHED}/ still holds the last good version; "
              f"the rejected tables are in {STAGING}/ for inspection.")
        sys.exit(1)

    publish(STAGING, PUBLISHED)
    LAST_GOOD.write_text(json.dumps({"finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "counts": {"patient_records": len(clean_t["patient_records"]),
                                                "golden_patients": len(clean_t["golden_patients"]),
                                                "encounters": len(clean_t["encounters"])}}, indent=2))
    record_run("PUBLISHED", None, all_checks, started)
    warns = sum(c.status == "WARN" for c in all_checks)
    print(f"\nPublished to {PUBLISHED}/  ({len(all_checks)} checks, {warns} warnings to review)")


if __name__ == "__main__":
    main()
