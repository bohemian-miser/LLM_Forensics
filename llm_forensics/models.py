"""Register local base models without copying them or importing GPU libraries."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def registry_path() -> Path:
    root = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
    return root / 'llm_forensics' / 'models.json'


def read_registry() -> dict[str, str]:
    p = registry_path()
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError(f'{p}: expected model-id to local-directory mappings')
    return data


def local_model(name: str) -> str:
    target = read_registry().get(name)
    if target is None:
        return name
    path = Path(target)
    if not (path / 'config.json').is_file():
        raise FileNotFoundError(f'Registered base {name!r} is unavailable at {target}; '
                                'register its current directory with llmf models register')
    return str(path)


def register(name: str, directory: str) -> None:
    path = Path(directory).expanduser().resolve()
    if not (path / 'config.json').is_file() or not list(path.glob('*.safetensors')):
        raise ValueError(f'{path}: expected an HF model directory with config.json and safetensors weights')
    data = read_registry()
    data[name] = str(path)
    dest = registry_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=dest.parent, delete=False) as f:
        tmp = Path(f.name)
        json.dump(data, f, indent=2)
    try:
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    reg = sub.add_parser('register', help='use an existing local directory for a base model ID')
    reg.add_argument('model_id')
    reg.add_argument('directory')
    sub.add_parser('list', help='show registered model directories')
    a = ap.parse_args(argv)
    if a.command == 'register':
        register(a.model_id, a.directory)
        print(f'{a.model_id} -> {local_model(a.model_id)}')
    else:
        for name, path in sorted(read_registry().items()):
            print(f'{name} -> {path}')
    return 0
