"""Combine separate per-model fresh runs without modifying frozen public records."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from scripts.audit_release import audit_records


def merge(inputs,destination):
    destination=Path(destination).resolve()
    if not destination.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Write merged runs under outputs/')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Choose a new empty output directory')
    batches={};counts={}
    for name in ('ratings.jsonl','recognition.jsonl'):
        rows=[];seen=set()
        for directory in inputs:
            for line in (Path(directory)/name).open():
                if not line.strip():continue
                r=json.loads(line)
                key=(r['leader_id'],r['model'],r.get('condition'),r.get('framing'),r.get('repeat'),r.get('speech_id'))
                if key in seen:raise ValueError('Duplicate associated record across fresh runs')
                seen.add(key);rows.append(r)
        batches[name]=rows;counts[name]=len(rows)
    if counts!={'ratings.jsonl':29625,'recognition.jsonl':960}:
        raise ValueError('Complete all five models, evidence conditions, framings and recognition probes before merging')
    # Totals alone cannot establish that every executed cell is present. Validate
    # the full schema/grid before creating or writing the user's destination.
    # Fresh recognition guesses may have any independently audited change count.
    with tempfile.TemporaryDirectory(prefix='personality-merge-') as temporary:
        candidate=Path(temporary)
        for name,rows in batches.items():
            (candidate/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        audit_records(ROOT,records_dir=candidate,expected_label_changes=None)
    destination.mkdir(parents=True,exist_ok=True)
    for name,rows in batches.items():
        (destination/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    return counts

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs',nargs='+',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();print(merge(a.inputs,a.output))
