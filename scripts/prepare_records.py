"""Adapt the public, allowlisted frozen records to the executed analysis schemas.

All generated files stay under ignored outputs/. Per-record labels travel with their own
recognition responses; fresh runs never inherit the original correction table.
"""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
FRAMING = {'E':'G', 'neutral':'O', 'P':'K'}


def prepare(root=ROOT, records_dir=None):
    root=Path(root)
    records_dir=Path(records_dir) if records_dir else root/'records'
    from scripts.audit_release import audit_records
    audit_records(root,records_dir,expected_label_changes=None)
    roster={r['leader_id']:r for r in csv.DictReader((root/'manifests/sample.csv').open())}
    speeches={r['speech_id']:r for r in csv.DictReader((root/'manifests/speeches.csv').open())}
    records=[json.loads(l) for l in (records_dir/'ratings.jsonl').open() if l.strip()]
    if len(records)!=29625:
        raise ValueError('Analysis requires the complete 29,625-record five-model rating matrix; finish all model runs first.')
    paths={
        ('name-context',False):'results_20260705T223912.jsonl',
        ('name-context',True):'results_openrouter_gpt-5.5_20260706.jsonl',
        ('speech',False):'results_speech_20260706T131201.jsonl',
        ('speech',True):'results_speech_openrouter_gpt-5.5_20260716T131331.jsonl',
    }
    out=root/'outputs';out.mkdir(exist_ok=True)
    batches={v:[] for v in paths.values()}
    batches.update({'name_arm_bridge.jsonl':[], 'name_arm_bridge_gpt-5.5.jsonl':[]})
    for r in records:
        if r['condition'] not in ('name-context','named-speech','anon-speech'):continue
        closed=r['model']=='gpt-5.5'; speech=r['speech_id'] is not None
        row={**r,'framing':FRAMING[r['framing']],'parse_ok':True,
             'kind':'rate','name_arm_id':r['leader_id'],'in_name_arm':True,
             'prompt_tokens':0}
        batches[paths[('speech' if speech else 'name-context',closed)]].append(row)
        if not speech and int(roster[r['leader_id']]['is_bridge']):
            batches['name_arm_bridge_gpt-5.5.jsonl' if closed else 'name_arm_bridge.jsonl'].append(row)
    for closed in (False,True):
        rows=[]
        for r in map(json.loads,(records_dir/'recognition.jsonl').open()):
            if (r['model']=='gpt-5.5')!=closed:continue
            person=roster[r['leader_id']]
            rows.append({**r,'name_arm_id':r['leader_id'],'in_name_arm':True,
                         'reid':r['guess'],'reid_score':r['auto_label'],
                         'true_name':person['full_name'],'true_country':person['country'],
                         'language':speeches[r['speech_id']]['language']})
        batches['reid_speech_openrouter_gpt-5.5_20260716T131331.jsonl' if closed else 'reid_speech_20260706T131201.jsonl']=rows
    for name,rows in batches.items():
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (out/'metrics').mkdir(exist_ok=True)
    return {k:len(v) for k,v in batches.items()}

if __name__=='__main__':
    print(json.dumps(prepare(),indent=2))
