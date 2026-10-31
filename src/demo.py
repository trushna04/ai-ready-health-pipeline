"""
demo.py  (Phase 5)

The whole project in one command, from messy raw data to a data card:

  1. generate_messy_data.py   messy synthetic data + answer key
  2. run_pipeline.py          clean, match, gate, de-identify, gate, publish
  3. access.py                role-based access demo, writes the audit log
  4. evaluate_phase1.py       grade patient matching against the answer key
  5. evaluate_phase2.py       PHI leak scan, re-identification risk, tamper test
  6. demo_gate_failures.py    break the data 9 ways, prove each gate blocks it
  7. build_ai_dataset.py      features, patient-level split, labels
  8. evaluate_phase4.py       baselines that test the eval set itself
  9. data_card.py             docs/data_card.md

Each step runs as its own process, exactly as you would run it by hand, so the
demo can't hide anything the individual scripts do. Full output of every step
goes to data/demo_log.txt; the screen shows only a summary.

If any step fails, the demo stops there and says which step and where to look.

Run:
  python src/demo.py              (seed 42, the numbers in the README)
  python src/demo.py --seed 7     (a different dataset; the story should hold)
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import demo_gate_failures

DATA = Path("data")
LOG = DATA / "demo_log.txt"


def run_step(number, total, title, command):
    """Run one script, append its output to the log, stop the demo if it fails."""
    started = time.time()
    print(f"[{number}/{total}] {title:<46}", end="", flush=True)
    result = subprocess.run([sys.executable, *command], capture_output=True, text=True)
    with open(LOG, "a") as log:
        log.write(f"\n{'=' * 70}\n$ python {' '.join(command)}\n{'=' * 70}\n")
        log.write(result.stdout + result.stderr)
    if result.returncode != 0:
        print("FAILED")
        print(f"\nStep {number} ({command[0]}) exited with code {result.returncode}. "
              f"Its output is at the end of {LOG}.")
        sys.exit(result.returncode)
    print(f"ok  ({time.time() - started:.1f}s)")


def load(path):
    return json.loads(Path(path).read_text())


def summary():
    """Read the numbers each step saved and print the story in a few lines."""
    m1 = load("data/eval/phase1_metrics.json")
    m2 = load("data/eval/phase2_metrics.json")
    manifest = load("data/answer_key/mess_manifest.json")
    gates = [load(f"data/quality/gate_report_{g}.json") for g in ("clean", "deid", "ai_ready")]
    n_checks = sum(len(g["checks"]) for g in gates)
    n_warn = sum(c["status"] == "WARN" for g in gates for c in g["checks"])
    ai = load("data/ai_ready/manifest.json")
    ev = load("data/ai_ready/eval_results.json")
    acc = ev["note_task"]["accuracy"]
    # demo_gate_failures.py exits with an error unless EVERY fault is blocked,
    # so reaching this point means all of them were.
    n_faults = len(demo_gate_failures.FAULTS)

    pipe, naive = m1["pipeline"], m1["baseline_last_dob_phone"]
    notes = m2["note_scrubbing"]
    k_year, k_band = m2["k_anonymity_if_birth_year_published"], m2["k_anonymity_published"]

    lines = [
        "",
        "=" * 70,
        "  RESULTS",
        "=" * 70,
        f"Raw data      {manifest['raw_patient_rows']} patient rows, {manifest['raw_encounter_rows']} visits, "
        f"{manifest['true_people']} real people behind them",
        "",
        "Matching      who is the same person?",
        f"              naive rule (last name + DOB + phone): {naive['wrong_merges']} wrong merges (twins)",
        f"              this pipeline: {pipe['wrong_merges']} wrong merges, "
        f"precision {pipe['precision']:.3f}, recall {pipe['recall']:.3f}",
        f"              {m1['review_queue']['pairs']} uncertain pairs sent to a human instead of guessed",
        "",
        "Governance    PHI removed before anyone downstream sees the data",
        f"              notes with PHI typed in: {notes['notes_with_leaked_phi']}, scrubbed: {notes['caught']}, "
        f"still leaking: {notes['still_contain_phi_after_scrub']}",
        f"              patients one-of-a-kind: {k_year['unique_k1']} if birth year published (allowed) "
        f"-> {k_band['unique_k1']} with age band",
        f"              audit log tampering detected: {m2['audit_log']['tampered_copy_detected']}",
        "",
        f"Quality       {n_checks} checks across 3 gates, 0 failures, {n_warn} warnings to review",
        f"              fault injection: {n_faults} / {n_faults} faults blocked",
        "",
        f"AI-ready      {ai['visits']} visits, split by patient: "
        f"{ai['patients']['train']} train / {ai['patients']['eval']} eval",
        f"              'first diagnosis word' rule on negated notes: {acc['first_keyword_mentioned']['negation']:.0%}"
        f" (fooled by 'No hx of diabetes')",
        f"              memorising note text scores {acc['memorizer (exact text lookup)']['all']:.0%}: "
        f"eval set too easy, and the data card says so",
        f"              real people on both sides via unresolved duplicates: "
        f"{ev['cross_split_leakage']['real_people_in_both_splits']}",
        "",
        "Read next     docs/data_card.md                   what the AI data can and can't be trusted for",
        "              data/quality/quality_report_*.md    what every gate checked",
        f"              {str(LOG):<34} full output of every step",
    ]
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run the whole project end to end.")
    parser.add_argument("--seed", type=int, default=42, help="which synthetic dataset to generate")
    args = parser.parse_args()

    # Start clean: old outputs from another seed would mix with the new ones.
    shutil.rmtree(DATA, ignore_errors=True)
    DATA.mkdir()
    LOG.write_text(f"Demo run, seed {args.seed}\n")

    steps = [
        ("Generate messy synthetic data", ["src/generate_messy_data.py", "--seed", str(args.seed)]),
        ("Clean, match, gate, de-identify, publish", ["src/run_pipeline.py"]),
        ("Role-based access + audit log", ["src/access.py"]),
        ("Grade patient matching (answer key)", ["src/evaluate_phase1.py"]),
        ("Check for PHI leaks and re-identification", ["src/evaluate_phase2.py"]),
        ("Break the data 9 ways, confirm gates block", ["src/demo_gate_failures.py"]),
        ("Build AI-ready dataset (own gate)", ["src/build_ai_dataset.py"]),
        ("Run baselines on the eval set", ["src/evaluate_phase4.py"]),
        ("Write the data card", ["src/data_card.py"]),
    ]
    print(f"AI-Ready Health Data Pipeline: end-to-end demo (seed {args.seed}, synthetic data only)\n")
    for i, (title, command) in enumerate(steps, start=1):
        run_step(i, len(steps), title, command)
    summary()


if __name__ == "__main__":
    main()
