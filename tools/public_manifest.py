"""Print the public source/demo package without touching confidential artifacts."""

import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES=['app.py','run.py','config.yaml','requirements.txt','pytest.ini','.gitignore','.env.example',
       'README.md','INTERVIEW_CHEATSHEET.md','PROGRESS.md','.streamlit/config.toml','config/hints.yaml','eval/README.md']
FOLDERS=['src','tests','demo','tools']
DEMO_SCREENSHOTS={'demo/screenshots/downloads.png','demo/screenshots/timeline.png'}


def public_files(root=ROOT):
    root=Path(root).resolve()
    paths=[root/name for name in FILES]
    for folder in FOLDERS:
        paths.extend((root/folder).rglob('*'))
    result=[]
    for path in paths:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            continue
        rel=path.relative_to(root)
        if any(part in {'__pycache__','.pytest_cache','.cache','.venv','data','parsed','outputs','.git'} for part in rel.parts):
            continue
        if path.suffix in {'.pyc','.pyo'} or (path.suffix == '.png' and rel.as_posix() not in DEMO_SCREENSHOTS):
            continue
        result.append(rel.as_posix())
    return sorted(set(result))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-private',action='store_true',help='Compare public text against long private screenplay passages; prints paths only')
    args=parser.parse_args()
    files=public_files()
    print('\n'.join(files))
    if args.audit_private:
        import json
        public={name:(ROOT/name).read_text(encoding='utf-8-sig') for name in files if name not in DEMO_SCREENSHOTS}
        passages=set()
        for path in (ROOT/'parsed').glob('episode_*.json'):
            episode=json.loads(path.read_text(encoding='utf-8'))
            passages.update(line['text'] for line in episode['lines'] if len(line['text'])>=50)
        matches=[name for name,text in public.items() if any(passage in text for passage in passages)]
        if matches:
            raise SystemExit('Private content found in: '+', '.join(matches))
        print(f'CONFIDENTIALITY PASS: {len(files)} public files; no long private screenplay passages found.')


if __name__=='__main__':
    main()
