"""
evaluate_phase4.py  (Phase 4)

Simple baselines, run on the eval sets, to prove the EVALUATION works before any
real model is trusted. A good eval set should:
  - make a naive model look bad on the hard cases (negations, empty notes),
  - expose shortcuts (a model that memorises note text should not get credit),
  - show when a task has no learnable signal at all.

Baselines are plain Python on purpose: the point is the eval, not the model.

It also uses the answer key for one thing the pipeline can't see: whether the
same REAL person ended up in both train and eval because two of their records
were never matched (they're sitting in the review queue).

Run (after build_ai_dataset.py):
  python src/evaluate_phase4.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from deidentify import pseudonym, secret_key

AI = Path("data/ai_ready")

# Keywords a non-expert might pick for each category, in a fixed order (ties go to the first).
KEYWORDS = {
    "hypertension": ["htn", "hypertension", "bp elevated", "elevated bp", "amlodipine", "lisinopril"],
    "type2_diabetes": ["t2dm", "diabetes", "dm2", "a1c", "metformin", "insulin"],
    "asthma": ["asthma", "wheezing", "albuterol", "inhaler"],
    "chest_pain": ["chest pain", "cp", "mi", "troponin", "ekg"],
    "respiratory_infection": ["uri", "influenza", "cough", "sore throat", "congestion", "fever"],
    "back_pain": ["back pain", "lbp", "radiculopathy"],
}

# A negation cue switches off everything up to the next comma or full stop:
# "No hx of diabetes. BP elevated today" -> " BP elevated today"
NEGATED_SPAN = re.compile(r"\b(?:no|denies|without)\b[^.,;]*", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Task A baselines: note -> diagnosis category
# ---------------------------------------------------------------------------

def keyword_scores(note):
    text = note.lower()
    return {cat: sum(bool(re.search(rf"\b{re.escape(w)}\b", text)) for w in words)
            for cat, words in KEYWORDS.items()}


def predict_first_mention(note):
    """
    The most common hand-written rule: whichever diagnosis word appears FIRST wins.
    'No hx of diabetes. BP elevated today' -> diabetes. Wrong, and that's the point.
    """
    if not isinstance(note, str):
        return None
    text = note.lower()
    first = {}
    for cat, words in KEYWORDS.items():
        positions = [m.start() for w in words for m in re.finditer(rf"\b{re.escape(w)}\b", text)]
        if positions:
            first[cat] = min(positions)
    return min(first, key=first.get) if first else None


def predict_keywords(note, handle_negation=False):
    """Pick the category with the most keyword hits. None if nothing matches."""
    if not isinstance(note, str):
        return None
    if handle_negation:
        note = NEGATED_SPAN.sub(" ", note)
    scores = keyword_scores(note)
    best = max(scores, key=scores.get)   # max() keeps the first one on ties
    return best if scores[best] > 0 else None


def make_memorizer(train):
    """Look up the exact note text in training. Scores well here, means nothing in real life."""
    lookup = train.groupby("diagnosis_note_scrubbed")["diagnosis_category"].agg(lambda s: s.mode()[0]).to_dict()
    return lambda note: lookup.get(note)


def accuracy(predictions, truth):
    """Share of rows predicted correctly. No prediction (None) counts as wrong."""
    return float((predictions == truth).mean()) if len(truth) else float("nan")


def evaluate_notes(train, eval_):
    majority = train["diagnosis_category"].mode()[0]
    memorize = make_memorizer(train)
    models = {
        "majority_class": lambda note: majority,
        "first_keyword_mentioned": predict_first_mention,
        "first_keyword, negation_aware": lambda note: predict_first_mention(
            NEGATED_SPAN.sub(" ", note) if isinstance(note, str) else note),
        "most_keyword_hits": lambda note: predict_keywords(note),
        "memorizer (exact text lookup)": memorize,
    }
    slices = {"all": pd.Series(True, index=eval_.index)}
    for tag in ["negation", "phi_redacted", "empty_note", "unseen_text"]:
        slices[tag] = eval_["eval_slices"].str.contains(tag)

    results = {}
    for name, model in models.items():
        pred = eval_["diagnosis_note_scrubbed"].map(model)
        results[name] = {s: round(accuracy(pred[mask], eval_.loc[mask, "diagnosis_category"]), 3)
                         for s, mask in slices.items()}
    sizes = {s: int(mask.sum()) for s, mask in slices.items()}
    return results, sizes


# ---------------------------------------------------------------------------
# Task B baselines: admission features -> length of stay
# ---------------------------------------------------------------------------

def mae(pred, truth):
    return round(float((pred - truth).abs().mean()), 2)


def evaluate_los(train, eval_):
    y_train = pd.to_numeric(train["length_of_stay_days"])
    y_eval = pd.to_numeric(eval_["length_of_stay_days"])
    inpatient = eval_["encounter_type"] == "inpatient"

    by_type = y_train.groupby(train["encounter_type"]).mean()
    inpatient_train = train["encounter_type"] == "inpatient"
    # Best effort with real features: within inpatients, mean stay by age band of 20 years.
    age_group = lambda df: (pd.to_numeric(df["age_at_admit"]) // 20).fillna(-1)
    by_age = y_train[inpatient_train].groupby(age_group(train[inpatient_train])).mean()

    models = {
        "overall_mean": pd.Series(y_train.mean(), index=eval_.index),
        "mean_by_encounter_type": eval_["encounter_type"].map(by_type),
        "type + inpatient mean by age group": eval_["encounter_type"].map(by_type).where(
            ~inpatient, age_group(eval_).map(by_age).fillna(by_type["inpatient"])),
    }
    results = {name: {"mae_all": mae(p, y_eval), "mae_inpatient": mae(p[inpatient], y_eval[inpatient])}
               for name, p in models.items()}
    sizes = {"all": len(eval_), "inpatient": int(inpatient.sum())}
    return results, sizes


# ---------------------------------------------------------------------------
# Hidden leakage: one real person on both sides
# ---------------------------------------------------------------------------

def cross_split_people(features):
    """
    The split guarantees no PATIENT KEY is on both sides. But if the matcher missed a
    duplicate, one real person has two patient keys, and those can land on different
    sides. Only the answer key can see this; it's the cost of the review queue.
    """
    key = secret_key()
    identity = pd.read_csv("data/answer_key/patient_identity.csv", dtype=str)
    records = pd.read_csv("data/clean/patient_records.csv", dtype=str)
    person_keys = (records.merge(identity, on="source_record_id")
                          .assign(patient_key=lambda d: [pseudonym(key, g, "PT") for g in d["golden_patient_id"]])
                          [["person_id", "patient_key"]].drop_duplicates())
    split_of = features.drop_duplicates("patient_key").set_index("patient_key")["split"]
    person_keys["split"] = person_keys["patient_key"].map(split_of)
    sides = person_keys.dropna().groupby("person_id")["split"].nunique()
    return {"real_people_with_visits": int(len(sides)), "real_people_in_both_splits": int((sides > 1).sum())}


# ---------------------------------------------------------------------------

def print_table(title, results, sizes, metric_label):
    """results: {model: {column: value}}; sizes: {column: row count} for the header."""
    columns = list(next(iter(results.values())).keys())
    print(f"\n=== {title} ===  ({metric_label})")
    header = "".join(f"{f'{c} (n={sizes[c]})':>22}" for c in columns)
    print(f"{'model':<34}{header}")
    for name, row in results.items():
        print(f"{name:<34}" + "".join(f"{v:>22}" for v in row.values()))


def main():
    note_train = pd.read_csv(AI / "note_classification/train.csv", dtype=str)
    note_eval = pd.read_csv(AI / "note_classification/eval.csv", dtype=str)
    los_train = pd.read_csv(AI / "length_of_stay/train.csv", dtype=str)
    los_eval = pd.read_csv(AI / "length_of_stay/eval.csv", dtype=str)
    features = pd.read_csv(AI / "features.csv", dtype=str)

    note_results, note_sizes = evaluate_notes(note_train, note_eval)
    los_results, los_sizes = evaluate_los(los_train, los_eval)
    leakage = cross_split_people(features)
    duplicate_text = float(note_eval["eval_slices"].str.contains("seen_in_train").mean())
    distinct_notes = int(features["diagnosis_note_scrubbed"].nunique())

    print_table("Task A: note -> diagnosis category", note_results, note_sizes, "accuracy, higher is better")
    print_table("Task B: length of stay", los_results, {"mae_all": los_sizes["all"], "mae_inpatient": los_sizes["inpatient"]},
                "mean absolute error in days, lower is better")
    print(f"\nEval notes whose exact text is also in training: {duplicate_text:.1%} "
          f"(only {distinct_notes} distinct notes across {len(features)} visits)")
    print(f"Real people with visits in BOTH splits (missed duplicates): "
          f"{leakage['real_people_in_both_splits']} of {leakage['real_people_with_visits']}")

    results = {"note_task": {"accuracy": note_results, "slice_sizes": note_sizes},
               "los_task": {"mae_days": los_results, "sizes": los_sizes},
               "eval_notes_seen_in_train": round(duplicate_text, 3),
               "distinct_notes": distinct_notes,
               "cross_split_leakage": leakage,
               "label_counts": dict(Counter(features["diagnosis_category"]))}
    (AI / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {AI / 'eval_results.json'}")


if __name__ == "__main__":
    main()
