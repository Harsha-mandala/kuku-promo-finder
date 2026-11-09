"""Keep uploads and secrets isolated to one editor session."""

import hashlib
import io
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from ..editorial_io import digest
from ..run_status import KEY_NAMES


def new_workspace():
    return tempfile.TemporaryDirectory(prefix='kuku_promo_ui_')


def clear_workspace(workspace):
    target=Path(workspace.name).resolve()
    if target.parent != Path(tempfile.gettempdir()).resolve() or not target.name.startswith('kuku_promo_ui_'):
        raise ValueError('Refusing to clear a directory outside this session workspace.')
    workspace.cleanup()


def unpack_uploads(files):
    """Read TXT/ZIP data without extracting arbitrary archive paths."""
    result={}
    def add(name, data):
        normalized=name.replace('\\','/')
        path=PurePosixPath(normalized)
        if path.is_absolute() or '..' in path.parts or ':' in normalized:
            raise ValueError('Upload contains an unsafe path.')
        if path.suffix.lower() != '.txt':
            return
        if len(data)>2_000_000:
            raise ValueError('An episode exceeds the 2 MB upload limit.')
        basename=path.name
        if basename.casefold() in {key.casefold() for key in result}:
            raise ValueError(f'Duplicate episode filename: {basename}')
        result[basename]=data.decode('utf-8-sig').encode('utf-8')
    for name,data in files:
        if name.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                infos=archive.infolist()
                if len(infos)>500 or sum(info.file_size for info in infos)>50_000_000:
                    raise ValueError('ZIP exceeds the upload size or file-count limit.')
                for info in infos:
                    if info.is_dir():
                        continue
                    if stat.S_ISLNK(info.external_attr >> 16):
                        raise ValueError('ZIP links are not supported.')
                    add(info.filename, archive.read(info))
        elif name.lower().endswith('.txt'):
            add(name,data)
        else:
            raise ValueError('Upload .txt files or one .zip input pack.')
    if not result or len(result)>200 or sum(map(len,result.values()))>50_000_000:
        raise ValueError('Upload 1–200 episode TXT files, at most 50 MB total.')
    return result


def upload_fingerprint(files):
    digest=hashlib.sha256()
    for name,data in sorted(files.items()):
        digest.update(name.encode())
        digest.update(data)
    return digest.hexdigest()


def write_uploads(root, files):
    target=Path(root)/'input'
    target.mkdir(parents=True,exist_ok=True)
    for name,data in files.items():
        (target/name).write_bytes(data)
    return target


def resolve_credentials(pasted, secrets, environ, providers, *, environment_is_hosted=False):
    keys={}
    hosted=False
    for provider in providers:
        name=KEY_NAMES.get(provider)
        if not name:
            continue
        if pasted.get(name):
            keys[name]=pasted[name]
        elif secrets.get(name):
            keys[name]=secrets[name]
            hosted=True
        elif environ.get(name):
            keys[name]=environ[name]
            hosted = hosted or environment_is_hosted
    return keys,hosted


def upload_run_fingerprint(files, models, selector, prompts):
    versions = [(path.name, hashlib.sha256(path.read_bytes()).hexdigest())
                for path in sorted(Path(prompts).glob('*.md'))]
    return digest([upload_fingerprint(files), models, selector, versions])


def upload_run_root(workspace, fingerprint, generation=0):
    return Path(workspace.name) / f'{fingerprint}-{generation}'
