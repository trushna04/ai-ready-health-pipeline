"""Tests for Phase 4: AI-ready outputs, on tiny hand-made tables. Run: pytest"""

import pandas as pd

import build_ai_dataset as ai
import evaluate_phase4 as ev


def test_split_is_by_patient_and_repeatable():
    assert ai.assign_split("PT-abc") == ai.assign_split("PT-abc")
    patients = pd.DataFrame({"patient_key": ["P1", "P2"], "age_band": ["30-39"] * 2, "zip3": ["193"] * 2,
                             "sex": ["F", "M"], "insurance": ["Aetna"] * 2, "n_source_records": ["1"] * 2})
    visits = pd.DataFrame({"encounter_key": ["E1", "E2", "E3"], "patient_key": ["P1", "P1", "P2"],
                           "admit_date_shifted": ["2025-01-01", "2025-03-01", "2025-02-01"],
                           "systolic_bp": [None, "120", "130"], "diastolic_bp": [None, "80", "85"],
                           "heart_rate": ["70", "72", None]})
    df = ai.build_features(patients, visits)
    assert df.groupby("patient_key")["split"].nunique().max() == 1      # never split one patient
    assert df["prior_visit_count"].tolist() == [0, 1, 0]
    assert df["n_vitals_missing"].tolist() == [2, 0, 1]


def test_split_share_is_close_to_target():
    splits = pd.Series([ai.assign_split(f"PT-{i}") for i in range(5000)])
    assert abs((splits == "eval").mean() - ai.EVAL_SHARE) < 0.02


def test_slices_tag_hard_cases():
    eval_ = pd.DataFrame({"diagnosis_note_scrubbed": ["Denies chest pain. Asthma refill.", None,
                                                      "Pt reachable at [PHONE].", "brand new text"]})
    tags = ai.tag_slices(eval_, train_notes={"Denies chest pain. Asthma refill."}).tolist()
    assert tags[0] == "negation;seen_in_train"
    assert tags[1] == "empty_note"
    assert "phi_redacted" in tags[2]
    assert tags[3] == "unseen_text"


def _checks(df, los_columns=None):
    los_train = pd.DataFrame(columns=los_columns or ["age_at_admit", "length_of_stay_days"])
    note = pd.DataFrame({"patient_key": ["P1"], "diagnosis_note_scrubbed": ["ok"], "diagnosis_category": ["asthma"]})
    records = pd.DataFrame({"first_name": ["Maria"], "last_name": ["Garcia"]})
    return {c.name: c for c in ai.ai_checks(df, note, note, los_train, records)}


def test_checks_block_patient_in_both_splits():
    df = pd.DataFrame({"patient_key": ["P1", "P1"], "split": ["train", "eval"],
                       "diagnosis_category": ["asthma", "asthma"]})
    assert _checks(df)["no_patient_in_both_splits"].status == "FAIL"


def test_checks_block_leaky_feature_and_missing_label():
    df = pd.DataFrame({"patient_key": ["P1"], "split": ["train"], "diagnosis_category": [None]})
    checks = _checks(df, los_columns=["discharge_date_shifted", "length_of_stay_days"])
    assert checks["no_leaky_features"].status == "FAIL"
    assert checks["every_visit_labeled"].status == "FAIL"


def test_first_mention_falls_for_negation_but_negation_aware_does_not():
    note = "No hx of diabetes. BP elevated today, start amlodipine."
    assert ev.predict_first_mention(note) == "type2_diabetes"            # wrong on purpose
    assert ev.predict_first_mention(ev.NEGATED_SPAN.sub(" ", note)) == "hypertension"
    assert ev.predict_keywords(None) is None                               # empty note: no guess
