"""Tests of publication boundaries and complete experimental response coverage."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_audit", ROOT / "scripts" / "audit_release.py")
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def test_public_response_matrix_is_complete():
    report = audit.audit_records(ROOT)
    assert report == {
        "politicians": 203,
        "bridge_politicians": 50,
        "speeches": 192,
        "ratings": 29625,
        "ratings_by_condition": {
            "anon-speech": 5760,
            "context-only": 3045,
            "name-context": 9135,
            "name-only": 3045,
            "named-speech": 8640,
        },
        "recognition_probes": 960,
        "recognition_label_changes": 52,
    }


def test_correction_manifest_matches_frozen_recognition_records():
    fields, corrections = audit.read_csv(ROOT / "manifests" / "recognition_corrections.csv")
    assert set(fields) == {"model", "speech_id", "auto_label", "label", "partial_kind"}
    expected = {
        (row["model"], row["speech_id"]): (
            row["auto_label"], row["label"], row["partial_kind"] or None
        )
        for row in corrections
    }
    actual = {
        (row["model"], row["speech_id"]): (
            row["auto_label"], row["label"], row["partial_kind"]
        )
        for row in audit.jsonl(ROOT / "records" / "recognition.jsonl")
        if row["auto_label"] != row["label"]
    }
    assert len(corrections) == len(expected) == 52
    assert actual == expected


def test_release_tree_has_no_detected_private_data():
    audit.audit_text(ROOT)


def test_manifest_matches_every_publication_file():
    report = audit.audit_manifest(ROOT)
    assert report["publication_files"] > 0
    assert report["publication_bytes"] > 0


def tampered_rows(monkeypatch, mutate):
    original = audit.jsonl

    def records(path):
        for number, row in enumerate(original(path)):
            if path.name == "ratings.jsonl" and number == 0:
                row = mutate(row)
            yield row

    monkeypatch.setattr(audit, "jsonl", records)


def test_added_expert_response_field_is_rejected(monkeypatch):
    def mutate(row):
        row["expert_score"] = 2.5
        return row

    tampered_rows(monkeypatch, mutate)
    with pytest.raises(audit.AuditError, match="unapproved fields"):
        audit.audit_records(ROOT)


def test_boolean_item_is_not_accepted_as_integer_rating(monkeypatch):
    def mutate(row):
        row["ratings"]["s1"] = True
        return row

    tampered_rows(monkeypatch, mutate)
    with pytest.raises(audit.AuditError, match="integers"):
        audit.audit_records(ROOT)


def test_wrong_seed_is_rejected(monkeypatch):
    def mutate(row):
        row["seed"] = 999
        return row

    tampered_rows(monkeypatch, mutate)
    with pytest.raises(audit.AuditError, match="seed"):
        audit.audit_records(ROOT)


def test_duplicate_cell_is_rejected(monkeypatch):
    original = audit.jsonl

    def records(path):
        for number, row in enumerate(original(path)):
            yield row
            if path.name == "ratings.jsonl" and number == 0:
                yield row

    monkeypatch.setattr(audit, "jsonl", records)
    with pytest.raises(audit.AuditError, match="duplicate experimental"):
        audit.audit_records(ROOT)


def test_missing_cell_is_rejected(monkeypatch):
    original = audit.jsonl

    def records(path):
        for number, row in enumerate(original(path)):
            if path.name == "ratings.jsonl" and number == 0:
                continue
            yield row

    monkeypatch.setattr(audit, "jsonl", records)
    with pytest.raises(audit.AuditError, match="coverage mismatch"):
        audit.audit_records(ROOT)


@pytest.mark.parametrize("content", [
    "sk-" + "x" * 40,
    "hf_" + "x" * 30,
    "/" + "home" + "/researcher/private/cache.json",
    "https://" + "private-account--model-server.modal.run",
])
def test_secret_and_private_path_detection(tmp_path, content):
    (tmp_path / "accidental.py").write_text(content, encoding="utf-8")
    with pytest.raises(audit.AuditError, match="detected"):
        audit.audit_text(tmp_path)


def test_environment_variable_names_are_allowed(tmp_path):
    (tmp_path / "runner.py").write_text(
        'key = os.environ["OPENROUTER_API_KEY"]\n', encoding="utf-8"
    )
    audit.audit_text(tmp_path)


def test_binary_dataset_is_rejected(tmp_path):
    (tmp_path / "source.dta").write_bytes(b"unapproved dataset")
    with pytest.raises(audit.AuditError, match="binary dataset"):
        audit.audit_text(tmp_path)


def test_aggregate_prediction_metric_is_allowed(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "baseline.csv").write_text(
        "trait,expert_prediction_r\nopenness,0.2\n", encoding="utf-8"
    )
    audit.audit_text(tmp_path)


def test_criterion_values_in_results_are_rejected(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "criterion.csv").write_text(
        "expert_openness_mean\n2.5\n", encoding="utf-8"
    )
    with pytest.raises(audit.AuditError, match="criterion columns"):
        audit.audit_text(tmp_path)


def make_manifest(root):
    content = b"public code\n"
    (root / "module.py").write_bytes(content)
    manifest = {"files": [{"path": "module.py", "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}]}
    (root / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_unlisted_file_is_rejected(tmp_path):
    make_manifest(tmp_path)
    (tmp_path / "forgotten_notes.md").write_text("private notes", encoding="utf-8")
    with pytest.raises(audit.AuditError, match="unlisted"):
        audit.audit_manifest(tmp_path)


def test_modified_public_file_is_rejected(tmp_path):
    make_manifest(tmp_path)
    (tmp_path / "module.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(audit.AuditError, match="mismatch"):
        audit.audit_manifest(tmp_path)


def test_runtime_dataset_is_not_in_publication_allowlist(tmp_path):
    make_manifest(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "source.dta").write_bytes(b"local only")
    assert audit.audit_manifest(tmp_path)["publication_files"] == 1


def test_symlinked_public_file_is_rejected(tmp_path):
    make_manifest(tmp_path)
    (tmp_path / "outside.py").write_text("private", encoding="utf-8")
    (tmp_path / "module.py").unlink()
    (tmp_path / "module.py").symlink_to(tmp_path / "outside.py")
    with pytest.raises(audit.AuditError, match="symlink"):
        audit.audit_manifest(tmp_path)
