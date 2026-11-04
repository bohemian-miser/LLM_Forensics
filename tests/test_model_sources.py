"""Model location registration and installed-package GCS downloads (no GPU required)."""
import json
from pathlib import Path

import pytest

from llm_forensics import gcs, models


def test_registered_model_reused_and_missing_path_fails(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'config.json').write_text('{}')
    (base / 'model.safetensors').write_bytes(b'weights')
    assert models.local_model('Qwen/base') == 'Qwen/base'
    assert models.main(['register', 'Qwen/base', str(base)]) == 0
    assert models.local_model('Qwen/base') == str(base)
    assert models.main(['list']) == 0
    (base / 'config.json').unlink()
    with pytest.raises(FileNotFoundError, match='Registered base'):
        models.local_model('Qwen/base')


def test_register_rejects_incomplete_directory(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'config'))
    with pytest.raises(ValueError, match='HF model directory'):
        models.register('Qwen/base', str(tmp_path))
    assert not models.registry_path().exists()


def test_packaged_gcs_download_and_existing_file_reuse(tmp_path, monkeypatch):
    weights = b'weights'
    items = [{'name': 'edits/s-1/weights/delta.safetensors', 'size': len(weights), 'generation': '7'},
             {'name': 'edits/s-1/weights/edit_vectors.safetensors', 'size': 123}]
    monkeypatch.setattr(gcs, 'list_objects', lambda b, p: items)
    calls = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size):
            if getattr(self, 'done', False): return b''
            self.done = True
            return weights

    def open_url(url):
        calls.append(url)
        return Response()

    monkeypatch.setattr(gcs.urllib.request, 'urlopen', open_url)
    args = ['edits/s-1/weights', '--bucket', 'test', '--dest', str(tmp_path),
            '--exclude', 'edit_vectors']
    assert gcs.main(args) == 0
    out = tmp_path / 'weights'
    assert (out / 'delta.safetensors').read_bytes() == weights
    assert not (out / 'edit_vectors.safetensors').exists()
    assert json.loads((out / 'provenance.json').read_text())['source'] == 'gs://test/edits/s-1/weights/'
    assert gcs.main(args) == 0
    assert len(calls) == 1
    (out / 'delta.safetensors').write_bytes(b'bad')
    assert gcs.main(args) == 0
    assert len(calls) == 2
    assert (out / 'delta.safetensors').read_bytes() == weights
