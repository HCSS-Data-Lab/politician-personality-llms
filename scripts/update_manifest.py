"""Regenerate publication hashes after reviewing intentional source changes.

The static and schema audits must pass before a new allowlist is written.
"""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_release import audit,runtime_path


def update(root=ROOT):
    root=Path(root);audit(root,manifest=False)
    entries=[]
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if not p.is_file() or runtime_path(rel) or rel.as_posix()=='RELEASE_MANIFEST.json':continue
        entries.append({'path':rel.as_posix(),'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (root/'RELEASE_MANIFEST.json').write_text(json.dumps({'files':entries},indent=2)+'\n')
    return len(entries)

if __name__=='__main__':print(f'Updated hashes for {update()} publication files.')
