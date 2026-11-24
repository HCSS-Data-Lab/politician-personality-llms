"""A fresh-run merge must validate the full grid before creating its destination."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import merge_runs


@pytest.fixture
def merge_workspace(tmp_path,monkeypatch):
    (tmp_path/'manifests').mkdir()
    for name in ('sample.csv','speeches.csv'):
        shutil.copyfile(ROOT/'manifests'/name,tmp_path/'manifests'/name)
    source=tmp_path/'input';source.mkdir()
    for name in ('ratings.jsonl','recognition.jsonl'):
        shutil.copyfile(ROOT/'records'/name,source/name)
    monkeypatch.setattr(merge_runs,'ROOT',tmp_path)
    return source,tmp_path/'outputs'/'combined'


def rewrite_first(path,mutate):
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    rows[0]=mutate(rows[0])
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')


def test_same_count_invalid_matrix_creates_no_destination(merge_workspace):
    source,destination=merge_workspace
    def mutate(row):
        row['model']='unexecuted-model'
        return row
    rewrite_first(source/'ratings.jsonl',mutate)
    with pytest.raises(ValueError,match='unknown rating'):
        merge_runs.merge([source],destination)
    assert not destination.exists()


def test_duplicate_cell_creates_no_destination(merge_workspace):
    source,destination=merge_workspace
    path=source/'ratings.jsonl'
    rows=path.read_text(encoding='utf-8').splitlines()
    rows[1]=rows[0]
    path.write_text('\n'.join(rows)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Duplicate associated'):
        merge_runs.merge([source],destination)
    assert not destination.exists()


def test_invalid_guess_creates_no_destination(merge_workspace):
    source,destination=merge_workspace
    def mutate(row):
        row['guess']['confidence']=2
        return row
    rewrite_first(source/'recognition.jsonl',mutate)
    with pytest.raises(ValueError,match='confidence'):
        merge_runs.merge([source],destination)
    assert not destination.exists()


def test_complete_matrix_merges_with_fresh_automatic_labels(merge_workspace):
    source,destination=merge_workspace
    path=source/'recognition.jsonl'
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    for row in rows:
        row['label']=row['auto_label']
        row['partial_kind']=None
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
    assert merge_runs.merge([source],destination)=={'ratings.jsonl':29625,'recognition.jsonl':960}
    for name in ('ratings.jsonl','recognition.jsonl'):
        actual=[json.loads(line) for line in (destination/name).read_text(encoding='utf-8').splitlines()]
        expected=[json.loads(line) for line in (source/name).read_text(encoding='utf-8').splitlines()]
        assert actual==expected
