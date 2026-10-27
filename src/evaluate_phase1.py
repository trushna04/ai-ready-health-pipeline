"""
evaluate_phase1.py  (Phase 1)

Grades the pipeline against the answer key. This is the ONLY Phase 1 file
allowed to read data/answer_key/. The pipeline never sees it.

Matching is graded on PAIRS of records:
  precision = of the record pairs we merged, how many really are one person
              (low precision = we mixed two people's charts: the dangerous error)
  recall    = of the record pairs that really are one person, how many we merged
              (low recall = leftover duplicates: annoying, but safe)

Two naive baselines are graded the same way, to show what the rules buy us.

Run (after pipeline.py):
  python src/evaluate_phase1.py
"""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

KEY, CLEAN, REVIEW = Path("data/answer_key"), Path("data/clean"), Path("data/review")


def same_cluster_pairs(record_to_cluster):
    """All unordered record pairs that share a cluster label."""
    groups = defaultdict(list)
    for rec, cluster in record_to_cluster.items():
        groups[cluster].append(rec)
    return {tuple(sorted(p)) for members in groups.values() for p in combinations(members, 2)}


def score(predicted, truth):
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else 1.0
    recall = tp / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"pairs_merged": len(predicted), "correct": tp, "wrong_merges": len(predicted) - tp,
            "missed": len(truth) - tp, "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4)}


def main():
    identity = pd.read_csv(KEY / "patient_identity.csv", dtype=str)
    labels = pd.read_csv(KEY / "encounter_labels.csv", dtype=str, keep_default_na=False)
    records = pd.read_csv(CLEAN / "patient_records.csv", dtype=str)
    raw = pd.read_csv("data/raw/patients.csv", dtype=str, keep_default_na=False).drop_duplicates()

    person_of = dict(zip(identity["source_record_id"], identity["person_id"]))
    truth = same_cluster_pairs({r: person_of[r] for r in records["source_record_id"]})
    results = {}

    # The pipeline.
    pipeline_pairs = same_cluster_pairs(dict(zip(records["source_record_id"], records["golden_patient_id"])))
    results["pipeline"] = score(pipeline_pairs, truth)

    # Baseline 1: exact match on raw first + last + DOB text. Misses every
    # nickname, typo, UPPERCASE name and differently formatted date.
    key1 = raw["first_name"] + "|" + raw["last_name"] + "|" + raw["dob"]
    results["baseline_exact_raw_text"] = score(
        same_cluster_pairs(dict(zip(raw["source_record_id"], key1))), truth)

    # Baseline 2: cleaned last name + DOB + phone, ignoring first name.
    # Looks reasonable, but merges twins (same last name, DOB and phone).
    rec = records.dropna(subset=["last_name", "dob", "phone"])
    key2 = rec["last_name"] + "|" + rec["dob"] + "|" + rec["phone"]
    results["baseline_last_dob_phone"] = score(
        same_cluster_pairs(dict(zip(rec["source_record_id"], key2))), truth)

    # Which wrong merges did the pipeline make, and which true pairs did it miss?
    wrong = sorted(pipeline_pairs - truth)
    missed = sorted(truth - pipeline_pairs)
    by_id = records.set_index("source_record_id")
    show = ["source_system", "first_name", "last_name", "dob", "sex", "phone", "zip"]

    # Review queue: how many of its pairs are truly the same person?
    review = pd.read_csv(REVIEW / "match_review_queue.csv", dtype=str)
    review_true = sum(person_of[a] == person_of[b] for a, b in zip(review["record_a"], review["record_b"]))
    review_pairs = {tuple(sorted(p)) for p in zip(review["record_a"], review["record_b"])}
    results["review_queue"] = {"pairs": len(review), "truly_same_person": int(review_true),
                               "missed_pairs_covered_by_queue": len(set(missed) & review_pairs)}

    # Golden patients vs true people.
    results["golden_patients"] = int(records["golden_patient_id"].nunique())
    results["true_people"] = int(identity["person_id"].nunique())

    # Encounter linkage: does each clean encounter point to the right person?
    enc = pd.read_csv(CLEAN / "encounters.csv", dtype=str).merge(labels, on="encounter_id")
    enc["linked_person"] = enc["source_record_id"].map(person_of)
    quarantined = pd.read_csv("data/quarantine/encounters.csv", dtype=str, keep_default_na=False)
    q = quarantined.merge(labels, on="encounter_id")
    orphan_q = q[q["quarantine_reason"] == "orphan_mrn"]
    results["encounters"] = {
        "clean_linked_to_correct_person": f"{(enc['linked_person'] == enc['person_id']).sum()} / {len(enc)}",
        "true_orphans_quarantined": f"{(orphan_q['person_id'] == '').sum()} / {(labels['person_id'] == '').sum()}",
        "real_patients_wrongly_called_orphan": int((orphan_q["person_id"] != "").sum()),
    }

    Path("data/eval").mkdir(exist_ok=True)
    Path("data/eval/phase1_metrics.json").write_text(json.dumps(results, indent=2))

    print("=== Patient matching (pairwise) ===")
    print(f"{'method':<26}{'merged':>8}{'wrong':>7}{'missed':>8}{'precision':>11}{'recall':>8}{'f1':>8}")
    for name in ("baseline_exact_raw_text", "baseline_last_dob_phone", "pipeline"):
        r = results[name]
        print(f"{name:<26}{r['pairs_merged']:>8}{r['wrong_merges']:>7}{r['missed']:>8}"
              f"{r['precision']:>11.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}")
    print(f"\nGolden patients: {results['golden_patients']}   true people: {results['true_people']}")
    print(f"Review queue: {results['review_queue']}")
    print(f"Encounters: {results['encounters']}")

    if wrong:
        print(f"\n--- {len(wrong)} wrong merges ---")
        for a, b in wrong:
            print(by_id.loc[[a, b], show].to_string(), "\n")
    print(f"\n--- {len(missed)} missed pairs (first 15) ---")
    for a, b in missed[:15]:
        print(by_id.loc[[a, b], show].to_string(), "\n")


if __name__ == "__main__":
    main()
