"""Clean-first-run and frozen/fresh isolation regressions."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from scripts.prepare_records import prepare
import speech_metrics as sm


def setup_root(tmp_path):
    shutil.copytree(ROOT/'records',tmp_path/'records')
    (tmp_path/'manifests').mkdir()
    for name in ('sample.csv','speeches.csv'):
        shutil.copyfile(ROOT/'manifests'/name,tmp_path/'manifests'/name)


def test_first_run_includes_closed_model_without_preexisting_outputs(tmp_path):
    setup_root(tmp_path)
    assert not (tmp_path/'outputs').exists()
    counts=prepare(tmp_path)
    assert counts['results_openrouter_gpt-5.5_20260706.jsonl']==1827
    assert counts['results_speech_openrouter_gpt-5.5_20260716T131331.jsonl']==2880
    assert counts['reid_speech_openrouter_gpt-5.5_20260716T131331.jsonl']==192
    assert counts['name_arm_bridge_gpt-5.5.jsonl']==450
    assert sm.GPT_SPEECH_FILE in sm.SPEECH_FILES
    assert sm.GPT_REID_FILE in sm.REID_FILES
    assert sm.ANCHOR in sm.ANCHORS


def test_fresh_guesses_never_inherit_historical_corrections(tmp_path,monkeypatch):
    setup_root(tmp_path)
    rows=[json.loads(l) for l in (tmp_path/'records/recognition.jsonl').open()]
    target=next(r for r in rows if r['label']!=r['auto_label'])
    target['label']=target['auto_label']
    (tmp_path/'records/recognition.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    prepare(tmp_path)
    monkeypatch.setattr(sm,'REID_FILES',[
        tmp_path/'outputs/reid_speech_20260706T131201.jsonl',
        tmp_path/'outputs/reid_speech_openrouter_gpt-5.5_20260716T131331.jsonl'])
    labels,_=sm.load_recognition()
    assert labels[(target['speech_id'],target['model'])]==target['auto_label']
