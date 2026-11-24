import json
from pathlib import Path
import sys
import tarfile
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import package_release


def test_archive_uses_allowlist_even_with_local_external_data(tmp_path,monkeypatch):
    (tmp_path/'README.md').write_text('public code')
    (tmp_path/'data').mkdir();(tmp_path/'data/criterion.csv').write_text('private local input')
    (tmp_path/'.env').write_text('never package this')
    (tmp_path/'RELEASE_MANIFEST.json').write_text(json.dumps({'files':[{'path':'README.md'}]}))
    monkeypatch.setattr(package_release,'audit',lambda root: {})
    path=tmp_path/'outputs/release.tar.gz'
    assert package_release.package(path,tmp_path)==2
    with tarfile.open(path) as archive:
        assert set(archive.getnames())=={'politician-personality-llms/README.md','politician-personality-llms/RELEASE_MANIFEST.json'}


def test_packager_requires_runtime_destination(tmp_path):
    with pytest.raises(ValueError,match='below outputs'):
        package_release.package(tmp_path/'release.tar.gz',tmp_path)
