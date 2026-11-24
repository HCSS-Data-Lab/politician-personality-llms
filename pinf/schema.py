# Adapted from the executed study code; see docs/provenance.md.
from __future__ import annotations
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from .config import OCCASION_TYPES, TRAITS

@dataclass
class ExpertScore:
    mean: float
    sd: Optional[float]
    n_raters: int

@dataclass
class ManifestRow:
    leader_id: str
    full_name: str
    country: str
    office: str
    term_start: Optional[str]
    term_end: Optional[str]
    speeches: dict[str, str]
    expert: dict[str, ExpertScore]

@dataclass
class RatingRecord:
    leader_id: str
    speech_id: Optional[str]
    occasion: Optional[str]
    condition: str
    model: str
    repeat: int
    seed: Optional[int]
    item_ratings: dict[str, int]
    trait_scores: dict[str, float]
    response_date: Optional[str]
    provider: Optional[str]
    raw_response: str
_MANIFEST_BASE_COLS = ['leader_id', 'full_name', 'country', 'office', 'term_start', 'term_end']
_SPEECH_COLS = [f'speech_{occ}' for occ in OCCASION_TYPES]
_EXPERT_COLS = [col for trait in TRAITS for col in (f'expert_{trait}_mean', f'expert_{trait}_sd', f'expert_{trait}_n')]
MANIFEST_FIELDNAMES = _MANIFEST_BASE_COLS + _SPEECH_COLS + _EXPERT_COLS

def _opt_str(value: Optional[str]) -> str:
    return '' if value is None else str(value)

def manifest_to_csv(rows: list[ManifestRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            record = {'leader_id': row.leader_id, 'full_name': row.full_name, 'country': row.country, 'office': row.office, 'term_start': _opt_str(row.term_start), 'term_end': _opt_str(row.term_end)}
            for occ in OCCASION_TYPES:
                record[f'speech_{occ}'] = row.speeches.get(occ, '')
            for trait in TRAITS:
                score = row.expert.get(trait)
                if score is None:
                    record[f'expert_{trait}_mean'] = ''
                    record[f'expert_{trait}_sd'] = ''
                    record[f'expert_{trait}_n'] = ''
                else:
                    record[f'expert_{trait}_mean'] = score.mean
                    record[f'expert_{trait}_sd'] = '' if score.sd is None else score.sd
                    record[f'expert_{trait}_n'] = score.n_raters
            writer.writerow(record)

def manifest_from_csv(path: str | Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with Path(path).open('r', newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            speeches = {occ: raw[f'speech_{occ}'] for occ in OCCASION_TYPES if raw.get(f'speech_{occ}', '') != ''}
            expert: dict[str, ExpertScore] = {}
            for trait in TRAITS:
                mean_cell = raw.get(f'expert_{trait}_mean', '')
                if mean_cell == '':
                    continue
                sd_cell = raw.get(f'expert_{trait}_sd', '')
                n_cell = raw.get(f'expert_{trait}_n', '')
                expert[trait] = ExpertScore(mean=float(mean_cell), sd=None if sd_cell == '' else float(sd_cell), n_raters=int(float(n_cell)) if n_cell != '' else 0)
            rows.append(ManifestRow(leader_id=raw['leader_id'], full_name=raw['full_name'], country=raw['country'], office=raw['office'], term_start=None if raw.get('term_start', '') == '' else raw['term_start'], term_end=None if raw.get('term_end', '') == '' else raw['term_end'], speeches=speeches, expert=expert))
    return rows

def _record_to_dict(record: RatingRecord) -> dict:
    return asdict(record)

def _record_from_dict(data: dict) -> RatingRecord:
    return RatingRecord(**data)

def records_to_jsonl(records: list[RatingRecord], path: str | Path, *, append: bool=False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if append else 'w'
    with path.open(mode, encoding='utf-8') as fh:
        for record in records:
            fh.write(json.dumps(_record_to_dict(record), ensure_ascii=False) + '\n')

def append_record(path: str | Path, record: RatingRecord) -> None:
    records_to_jsonl([record], path, append=True)

def records_from_jsonl(path: str | Path) -> list[RatingRecord]:
    records: list[RatingRecord] = []
    with Path(path).open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(_record_from_dict(json.loads(line)))
    return records
