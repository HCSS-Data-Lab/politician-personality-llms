"""Reconstruct the executed speech inputs from separately downloaded transcripts.

Recipes contain source-character offsets and generated editing instructions, not
transcript bodies. Source and reconstructed hashes must match before files write.
"""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def apply_steps(source,steps):
    pieces=[]
    for op in steps:
        if set(op)=={'copy'}:
            start,end=op['copy']
            if type(start) is not int or type(end) is not int or not 0<=start<=end<=len(source):
                raise ValueError('Invalid source offset')
            pieces.append(source[start:end])
        elif set(op)=={'insert'} and isinstance(op['insert'],str):
            pieces.append(op['insert'])
        else:raise ValueError('Invalid transformation instruction')
    return ''.join(pieces)


def prepare(raw_dir,recipe_path=ROOT/'manifests/preprocessing.json',out_dir=ROOT/'data/speeches'):
    raw_dir,out_dir=Path(raw_dir),Path(out_dir)
    recipes=json.loads(Path(recipe_path).read_text())
    prepared=[]
    for r in recipes:
        sid=r['speech_id']
        if Path(sid).name!=sid or not sid.endswith('.txt'):
            raise ValueError('Unsafe speech identifier')
        source=(raw_dir/sid).read_text(encoding='utf-8-sig',errors='replace')
        if digest(source)!=r['source_sha256']:
            raise ValueError(f'{sid}: source differs from the required historical corpus version')
        clean=apply_steps(source,r['clean_steps'])
        anon=apply_steps(clean,r['anon_steps'])
        if digest(clean)!=r['clean_sha256'] or digest(anon)!=r['anon_sha256']:
            raise ValueError(f'{sid}: reconstructed input hash mismatch')
        prepared.append((sid,clean,anon))
    for variant in ('clean','anon'):(out_dir/variant).mkdir(parents=True,exist_ok=True)
    for sid,clean,anon in prepared:
        (out_dir/'clean'/sid).write_text(clean)
        (out_dir/'anon'/sid).write_text(anon)
    return len(prepared)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-dir',type=Path,default=ROOT/'data/raw_speeches')
    ap.add_argument('--recipes',type=Path,default=ROOT/'manifests/preprocessing.json')
    ap.add_argument('--out-dir',type=Path,default=ROOT/'data/speeches')
    a=ap.parse_args()
    print(f'Reconstructed {prepare(a.raw_dir,a.recipes,a.out_dir)} clean and scrubbed speech inputs locally.')

if __name__=='__main__':main()
