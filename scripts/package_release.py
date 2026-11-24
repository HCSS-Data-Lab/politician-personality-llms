"""Create a release archive from the audited publication allowlist only."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_release import audit


def package(destination,root=ROOT):
    root=Path(root);destination=Path(destination).resolve()
    # Archives are runtime outputs and never part of the publication candidate tree.
    if not destination.is_relative_to((root/'outputs').resolve()):
        raise ValueError('Write archives below outputs/ to keep them out of the publication tree')
    audit(root)
    entries=json.loads((root/'RELEASE_MANIFEST.json').read_text())['files']
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(destination,'w:gz') as archive:
        for name in ['RELEASE_MANIFEST.json']+[e['path'] for e in entries]:
            archive.add(root/name,arcname='politician-personality-llms/'+name,recursive=False)
    return len(entries)+1

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/politician-personality-llms.tar.gz')
    a=ap.parse_args();print(f'Packaged {package(a.output)} audited files.')
