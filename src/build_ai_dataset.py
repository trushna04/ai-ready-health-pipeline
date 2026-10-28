"""
build_ai_dataset.py  (Phase 4)

Turns the PUBLISHED de-identified tables into datasets a model can train and be
evaluated on. It never reads data/clean/ or anything with PHI in it, except the
PHI leak scan (which needs the real names to look for them).

Outputs (data/ai_ready/):
  features.csv                     one row per visit: every model-ready feature,
                                   the split, and the label
  note_classification/train.csv    task A: scrubbed note -> diagnosis category
  note_classification/eval.csv       (eval rows are tagged with "slices": hard cases)
  length_of_stay/train.csv         task B: what's known at admission -> days in hospital
  length_of_stay/eval.csv
  manifest.json                    counts, split sizes, check results

Two rules this file exists to enforce:
  1. Split by PATIENT, not by visit. If one patient's visits land on both sides,
     the model can recognise the patient instead of learning the task, and the
     eval score is inflated. That's called leakage.
  2. Only use what would be known at prediction time. Discharge date tells you
     the length of stay exactly, so it can never be a length-of-stay feature.

Where labels come from: the generator's answer key, which knows each visit's true
diagnosis. In a real hospital this step would be chart review by clinical coders,
or ICD-10 codes from billing. That's the only reason this file touches the answer
key, and it keeps nothing from it except encounter_key + label.

Run (after run_pipeline.py has published):
  python src/build_ai_dataset.py
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

import data_dictionary
import quality_gates as gates
from data_dictionary import check_columns
from deidentify import pseudonym, secret_key

PUBLISHED = Path("data/deidentified")
LABELS_SOURCE = Path("data/answer_key/encounter_labels.csv")
OUT = Path("data/ai_ready")

EVAL_SHARE = 0.20          # 20% of PATIENTS (not visits) go to eval
SPLIT_SALT = "split-v1"    # change it and you get a different, still repeatable, split
CATEGORIES = ["hypertension", "type2_diabetes", "asthma", "chest_pain",
              "respiratory_infection", "back_pain"]

# Task B may only use what is known when the patient is admitted.
LOS_FEATURES = ["age_at_admit", "sex", "insurance", "encounter_type", "source_system",
                "systolic_bp", "diastolic_bp", "heart_rate", "n_vitals_missing", "prior_visit_count"]
# Never allowed as features, and the build blocks if one sneaks in:
#   discharge_date_shifted  -> gives length of stay away exactly (target leakage)
#   admit_date_shifted      -> each patient is shifted by a different secret amount,
#                              so it means nothing across patients
LEAKY_COLUMNS = {"discharge_date_shifted", "admit_date_shifted", "length_of_stay_days"}

# Hard cases for the note classifier. A model that only matches keywords fails these.
NEGATION = re.compile(r"\b(denies|no hx|no\b|r/o)\b", re.IGNORECASE)
REDACTED = re.compile(r"\[(PHONE|MRN|NAME)\]")
VITALS = ["systolic_bp", "diastolic_bp", "heart_rate"]


# ---------------------------------------------------------------------------
# Step 1: features (one row per visit)
# ---------------------------------------------------------------------------

def assign_split(patient_key):
    """
    Deterministic split by patient: hash the patient key, look at the number.
    Same patient -> same split, every run, with no list to store.
    """
    bucket = int(hashlib.sha256(f"{SPLIT_SALT}|{patient_key}".encode()).hexdigest(), 16) % 100
    return "eval" if bucket < EVAL_SHARE * 100 else "train"


def build_features(patients, encounters):
    df = encounters.merge(patients[["patient_key", "age_band", "zip3", "sex", "insurance", "n_source_records"]],
                          on="patient_key", how="left", validate="many_to_one")

    df["n_vitals_missing"] = df[VITALS].isna().sum(axis=1)

    # How many earlier visits this patient had. Dates are shifted by the same
    # amount for every visit of one patient, so the ORDER within a patient is real.
    df["admit_date_shifted"] = pd.to_datetime(df["admit_date_shifted"])
    df["prior_visit_count"] = (df.groupby("patient_key")["admit_date_shifted"]
                                 .rank(method="min").astype(int) - 1)
    df["admit_date_shifted"] = df["admit_date_shifted"].dt.strftime("%Y-%m-%d")

    df["split"] = df["patient_key"].map(assign_split)
    return df


# ---------------------------------------------------------------------------
# Step 2: labels
# ---------------------------------------------------------------------------

def attach_labels(df, labels_path=LABELS_SOURCE):
    """
    Look up each visit's true category. The answer key uses the source encounter_id;
    we turn it into the same pseudonym deidentify.py used, then keep only
    (encounter_key, label). No source IDs survive this function.
    """
    key = secret_key()
    labels = pd.read_csv(labels_path, dtype=str)
    labels = pd.DataFrame({
        "encounter_key": [pseudonym(key, e, "EN") for e in labels["encounter_id"]],
        "diagnosis_category": labels["true_category"],
    })
    return df.merge(labels, on="encounter_key", how="left", validate="one_to_one")


def tag_slices(df, train_notes):
    """Semicolon-separated hard-case tags for each eval row."""
    def tags(note):
        if not isinstance(note, str):
            return "empty_note"
        out = []
        if NEGATION.search(note):
            out.append("negation")
        if REDACTED.search(note):
            out.append("phi_redacted")
        out.append("seen_in_train" if note in train_notes else "unseen_text")
        return ";".join(out)
    return df["diagnosis_note_scrubbed"].map(tags)


# ---------------------------------------------------------------------------
# Step 3: task files
# ---------------------------------------------------------------------------

def note_task(df):
    cols = ["encounter_key", "patient_key", "diagnosis_note_scrubbed", "diagnosis_category"]
    train = df[(df["split"] == "train") & df["diagnosis_note_scrubbed"].notna()][cols]  # nothing to learn from an empty note
    eval_ = df[df["split"] == "eval"][cols].copy()
    eval_["eval_slices"] = tag_slices(eval_, set(train["diagnosis_note_scrubbed"]))
    return train, eval_


def los_task(df):
    cols = ["encounter_key", "patient_key", *LOS_FEATURES, "length_of_stay_days"]
    return df[df["split"] == "train"][cols], df[df["split"] == "eval"][cols]


# ---------------------------------------------------------------------------
# Step 4: checks. Same Check/report machinery as the Phase 3 gates.
# ---------------------------------------------------------------------------

def ai_checks(df, note_train, note_eval, los_train, records):
    s = "ai_ready"
    Check = gates.Check
    train_patients = set(df.loc[df["split"] == "train", "patient_key"])
    eval_patients = set(df.loc[df["split"] == "eval", "patient_key"])
    overlap = train_patients & eval_patients
    eval_share = len(eval_patients) / (len(train_patients) + len(eval_patients))
    unlabeled = int(df["diagnosis_category"].isna().sum())
    unknown = sorted(set(df["diagnosis_category"].dropna()) - set(CATEGORIES))
    leaky = sorted(LEAKY_COLUMNS & set(los_train.columns[:-1]))   # last column is the target
    per_class = note_eval["diagnosis_category"].value_counts()
    thin = {c: int(per_class.get(c, 0)) for c in CATEGORIES if per_class.get(c, 0) < 20}

    checks = [
        Check(s, "no_patient_in_both_splits", "block", not overlap,
              f"{len(overlap)} patients in both train and eval"),
        Check(s, "every_visit_labeled", "block", unlabeled == 0 and not unknown,
              f"{unlabeled} visits without a label" + (f"; unknown labels {unknown}" if unknown else "")),
        Check(s, "no_leaky_features", "block", not leaky,
              f"leaky columns in length-of-stay features: {leaky}" if leaky else "only admission-time features"),
        gates.check_phi_leaks(s, {"note_train": note_train, "note_eval": note_eval}, records),
        Check(s, "eval_share", "warn", 0.15 <= eval_share <= 0.25,
              f"{eval_share:.1%} of patients in eval (target {EVAL_SHARE:.0%})"),
        Check(s, "eval_rows_per_category", "warn", not thin,
              f"categories with < 20 eval rows: {thin}" if thin else
              f"every category has >= 20 eval rows (min {int(per_class.min())})"),
    ]
    return checks


# ---------------------------------------------------------------------------

def main():
    if not (gates.QUALITY / "last_good_metrics.json").exists():
        sys.exit("No published data yet. Run `python src/run_pipeline.py` first: "
                 "the AI dataset is only ever built from data that passed every gate.")

    patients = pd.read_csv(PUBLISHED / "patients_deid.csv", dtype=str)
    encounters = pd.read_csv(PUBLISHED / "encounters_deid.csv", dtype=str)

    df = attach_labels(build_features(patients, encounters))
    note_train, note_eval = note_task(df)
    los_train, los_eval = los_task(df)

    # The leak scan needs the real names to search for them; it keeps nothing.
    records = pd.read_csv("data/clean/patient_records.csv", dtype=str)
    checks = ai_checks(df, note_train, note_eval, los_train, records)
    if not gates.report(checks, "ai_ready"):
        sys.exit("AI dataset NOT written. Fix the failed checks above.")

    features = df[["encounter_key", "patient_key", "split", "source_system", "encounter_type",
                   "admit_date_shifted", "age_at_admit", "age_band", "sex", "zip3", "insurance",
                   "n_source_records", "prior_visit_count", *VITALS, "n_vitals_missing",
                   "diagnosis_note_scrubbed", "length_of_stay_days", "diagnosis_category"]]
    outputs = {
        "ai_features": (OUT / "features.csv", features),
        "note_task_train": (OUT / "note_classification/train.csv", note_train),
        "note_task_eval": (OUT / "note_classification/eval.csv", note_eval),
        "los_task_train": (OUT / "length_of_stay/train.csv", los_train),
        "los_task_eval": (OUT / "length_of_stay/eval.csv", los_eval),
    }
    for name, (path, table) in outputs.items():
        check_columns(name, table.columns)   # nothing undocumented leaves the pipeline
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    manifest = {
        "visits": len(df),
        "patients": {"train": int((df.drop_duplicates("patient_key")["split"] == "train").sum()),
                     "eval": int((df.drop_duplicates("patient_key")["split"] == "eval").sum())},
        "note_task": {"train": len(note_train), "eval": len(note_eval)},
        "los_task": {"train": len(los_train), "eval": len(los_eval)},
        "label_counts": df["diagnosis_category"].value_counts().to_dict(),
        "eval_slices": note_eval["eval_slices"].str.split(";").explode().value_counts().to_dict(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    data_dictionary.main()   # docs/data_dictionary.md now includes the AI tables

    print(f"\nWrote {OUT}/")
    print(f"  patients  train {manifest['patients']['train']}, eval {manifest['patients']['eval']}")
    print(f"  note task train {len(note_train)}, eval {len(note_eval)}")
    print(f"  LOS task  train {len(los_train)}, eval {len(los_eval)}")
    print(f"  eval slices {manifest['eval_slices']}")


if __name__ == "__main__":
    main()
