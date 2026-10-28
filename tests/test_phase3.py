"""Tests for Phase 3 quality gates, on tiny hand-made tables. Run: pytest"""

import json

import pandas as pd

import quality_gates as g
from run_pipeline import publish


def test_rate_check_levels():
    limits = {"warn": 0.05, "block": 0.10}
    assert g._rate_check("s", "x", 0.01, limits, "").status == "PASS"
    assert g._rate_check("s", "x", 0.07, limits, "").status == "WARN"
    assert g._rate_check("s", "x", 0.20, limits, "").status == "FAIL"


def test_warnings_do_not_stop_the_pipeline():
    warn = g.Check("s", "a", "warn", False, "")
    fail = g.Check("s", "b", "block", False, "")
    assert g.passed([warn]) and not g.passed([warn, fail])


def test_reconciliation_catches_lost_rows():
    raw = pd.DataFrame({"id": ["1", "2", "3", "3"]})        # one exact duplicate
    assert g.check_reconciliation("s", raw, raw.iloc[:2], raw.iloc[2:3], "t").passed
    assert not g.check_reconciliation("s", raw, raw.iloc[:1], raw.iloc[2:3], "t").passed


def test_merge_conflict_alarm():
    ok = pd.DataFrame({"golden_patient_id": ["G1", "G1"], "formal_first_name": ["Robert", "Robert"],
                       "dob": ["1980-01-01", None]})        # blank DOB is not a conflict
    bad = ok.assign(formal_first_name=["Robert", "William"])  # twins merged
    assert g.check_merge_conflicts("s", ok).passed
    assert not g.check_merge_conflicts("s", bad).passed


def test_foreign_key():
    parent = pd.DataFrame({"k": ["A", "B"]})
    assert g.check_foreign_key("s", pd.DataFrame({"k": ["A"]}), "c", parent, "p", "k").passed
    assert not g.check_foreign_key("s", pd.DataFrame({"k": ["Z"]}), "c", parent, "p", "k").passed


def _deid(note):
    return {"patients_deid": pd.DataFrame({"patient_key": ["PT-1"], "age_band": ["30-39"]}),
            "encounters_deid": pd.DataFrame({"patient_key": ["PT-1"], "diagnosis_note_scrubbed": [note]})}


RECORDS = pd.DataFrame({"first_name": ["Maria"], "last_name": ["Garcia"]})


def test_phi_scan_passes_scrubbed_note():
    assert g.check_phi_leaks("s", _deid("LBP after lifting. Pt reachable at [PHONE]."), RECORDS).passed


def test_phi_scan_blocks_each_kind_of_leak():
    for note in ["Call 610-555-0142.", "See MRN A123456.", "Spouse is Maria Garcia.",
                 "Seen with Garcia.", "Linked to REC-000123."]:
        assert not g.check_phi_leaks("s", _deid(note), RECORDS).passed, note


def test_phi_scan_blocks_direct_identifier_column():
    deid = _deid("ok")
    deid["patients_deid"]["phone"] = "[removed]"            # column type alone is enough to block
    assert not g.check_phi_leaks("s", deid, RECORDS).passed


def test_drift_against_last_good_run(tmp_path):
    last = tmp_path / "last.json"
    assert g.check_drift("s", {"encounters": 100}, last).passed          # first run: nothing to compare
    last.write_text(json.dumps({"counts": {"encounters": 100}}))
    assert g.check_drift("s", {"encounters": 110}, last).passed
    assert g.check_drift("s", {"encounters": 60}, last).status == "WARN"


def test_strict_mode_blocks_dev_key(monkeypatch):
    monkeypatch.delenv("PIPELINE_SECRET_KEY", raising=False)
    assert g.check_secret_key("s", strict=False).status == "WARN"
    assert g.check_secret_key("s", strict=True).status == "FAIL"
    monkeypatch.setenv("PIPELINE_SECRET_KEY", "real-secret")
    assert g.check_secret_key("s", strict=True).passed


def test_publish_replaces_whole_folder(tmp_path):
    staging, published = tmp_path / "staging", tmp_path / "published"
    staging.mkdir(); published.mkdir()
    (staging / "new.csv").write_text("x")
    (published / "old.csv").write_text("y")
    publish(staging, published)
    assert [p.name for p in published.iterdir()] == ["new.csv"]
