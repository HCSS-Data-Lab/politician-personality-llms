"""Enforce the historical Wikimedia export's precise publication boundary."""
from pathlib import Path
import pytest
from scripts import audit_release as audit

ROOT = Path(__file__).resolve().parents[1]


def test_snapshot_matches_fixed_sample_and_source_crosswalk():
    assert audit.audit_prominence_snapshot(ROOT) == {'historical_prominence_rows': 203}


def tamper(monkeypatch, mutate):
    original = audit.read_csv
    def read(path):
        fields, rows = original(path)
        if path.name == 'prominence_2026-07-06.csv':
            mutate(fields, rows)
        return fields, rows
    monkeypatch.setattr(audit, 'read_csv', read)


def test_expert_column_cannot_be_added(monkeypatch):
    def mutate(fields, rows):
        fields.append('expert_openness_mean')
        for row in rows: row['expert_openness_mean'] = '2.5'
    tamper(monkeypatch, mutate)
    with pytest.raises(audit.AuditError, match='unapproved columns'):
        audit.audit_prominence_snapshot(ROOT)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unknown'])
def test_snapshot_membership_is_exact(monkeypatch, mutation):
    def mutate(fields, rows):
        if mutation == 'missing': rows.pop()
        elif mutation == 'duplicate': rows.append(dict(rows[0]))
        else: rows[0]['leader_id'] = 'outside_sample'
    tamper(monkeypatch, mutate)
    with pytest.raises(audit.AuditError, match='sample exactly'):
        audit.audit_prominence_snapshot(ROOT)


@pytest.mark.parametrize('column,value', [
    ('sitelinks', '-1'), ('sitelinks', '2.5'), ('sitelinks', 'NaN'),
    ('pageviews_election_year', ''), ('pageviews_election_year', 'inf'),
    ('pageviews_election_year', 'True'), ('pageviews_election_year', '1e5'),
])
def test_counts_are_complete_nonnegative_integers(monkeypatch, column, value):
    tamper(monkeypatch, lambda fields, rows: rows[0].update({column: value}))
    with pytest.raises(audit.AuditError, match='nonnegative integers'):
        audit.audit_prominence_snapshot(ROOT)


@pytest.mark.parametrize('column,value', [
    ('snapshot_date', '2026-11-24'), ('source_stratum', 'other'),
    ('qid', 'Q999999999'), ('enwiki_title', 'Different Person'),
    ('election_year', '1900'),
])
def test_historical_date_and_identifiers_are_not_silently_changed(monkeypatch, column, value):
    tamper(monkeypatch, lambda fields, rows: rows[0].update({column: value}))
    with pytest.raises(audit.AuditError):
        audit.audit_prominence_snapshot(ROOT)


def test_observed_zero_is_allowed(monkeypatch):
    tamper(monkeypatch, lambda fields, rows: rows[0].update(pageviews_election_year='0'))
    assert audit.audit_prominence_snapshot(ROOT)['historical_prominence_rows'] == 203
