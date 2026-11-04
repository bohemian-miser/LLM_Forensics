"""Collect agent-written detector candidates for review, without importing or running them.

    python -m lab.detector_candidates WORKDIR/traces --out WORKDIR/detector-review

The collector is rerunnable after every result collection. Original submissions remain in
traces; snapshots are content-addressed and never registered with llmf scan.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import tempfile
from pathlib import Path

FILES = ('manifest.json', 'detector.py', 'test_detector.py', 'README.md', 'validation.json')


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        f.write(data)
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def submission(source: Path, root: Path) -> tuple[dict, dict[str, bytes]]:
    source.resolve().relative_to(root.resolve())
    for p in (source, source.parent, source.parent.parent):
        if p.is_symlink():
            raise ValueError('candidate paths must not be symlinks')
    payload = {}
    for name in FILES:
        p = source / name
        if p.is_symlink() or not p.is_file():
            raise ValueError(f'missing regular file: {name}')
        payload[name] = p.read_bytes()
    manifest = json.loads(payload['manifest.json'])
    if manifest.get('schema_version') != 1:
        raise ValueError('manifest schema_version must be 1')
    if not re.fullmatch(r'[a-z][a-z0-9-]{1,63}', manifest.get('id', '')):
        raise ValueError('candidate id must be a lowercase slug')
    for key in ('summary', 'method', 'generalization'):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f'manifest needs {key}')
    if not isinstance(manifest.get('requires_base'), bool):
        raise ValueError('requires_base must be boolean')
    if not isinstance(manifest.get('known_limitations'), list):
        raise ValueError('known_limitations must be a list')
    provenance = manifest.get('provenance', {})
    if not isinstance(provenance, dict) or not provenance.get('source_suspect') or not provenance.get('reference'):
        raise ValueError('provenance needs source_suspect and reference')
    if provenance['source_suspect'] != source.parent.parent.name:
        raise ValueError('provenance source_suspect must match the collected trace directory')
    validation = json.loads(payload['validation.json'])
    if validation.get('status') not in ('untested', 'tested'):
        raise ValueError('validation status must be untested or tested')
    if not isinstance(validation.get('tests_run'), list):
        raise ValueError('validation needs tests_run (empty if untested)')
    if validation['status'] == 'tested' and not validation['tests_run']:
        raise ValueError('tested status needs tests_run evidence')
    if not isinstance(validation.get('resources'), dict):
        raise ValueError('validation needs resources (unknown measurements may be null)')
    tree = ast.parse(payload['detector.py'].decode('utf-8'))
    ast.parse(payload['test_detector.py'].decode('utf-8'))
    candidates = []
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        attrs = {t.id: n.value for n in cls.body if isinstance(n, ast.Assign)
                 for t in n.targets if isinstance(t, ast.Name)}
        if any(isinstance(base, ast.Name) and base.id == 'Detector' for base in cls.bases):
            candidates.append(attrs)
    if not any(isinstance(a.get('default'), ast.Constant) and a['default'].value is False
               and isinstance(a.get('id'), ast.Constant) and a['id'].value == manifest['id']
               for a in candidates):
        raise ValueError('detector.py needs a Detector subclass with matching id and default = False')
    return manifest, payload


def collect_candidates(traces: Path, out: Path) -> dict:
    """Snapshot valid submissions; index invalid ones too. Never execute candidate code."""
    traces, out = Path(traces), Path(out)
    if out.resolve().is_relative_to(traces.resolve()):
        raise ValueError('review output must be outside traces')
    records, errors = [], []
    for source in sorted(traces.glob('*/detector_candidates/*')):
        origin = str(source.relative_to(traces))
        try:
            manifest, payload = submission(source, traces)
            digest = hashlib.sha256()
            digest.update(origin.encode('utf-8'))
            for name, data in sorted(payload.items()):
                digest.update(name.encode('utf-8') + b'\0' + len(data).to_bytes(8, 'big') + data)
            fingerprint = digest.hexdigest()
            dest = out / manifest['id'] / fingerprint
            dest.mkdir(parents=True, exist_ok=True)
            for name, data in payload.items():
                target = dest / name
                if target.exists():
                    if target.read_bytes() != data:
                        raise ValueError(f'archived snapshot differs: {target}')
                else:
                    atomic_write(target, data)
            record = {'id': manifest['id'], 'summary': manifest['summary'], 'origin': origin,
                      'sha256': fingerprint, 'review_status': 'unreviewed',
                      'validation_claim': json.loads(payload['validation.json'])['status'],
                      'files': {n: hashlib.sha256(v).hexdigest() for n, v in payload.items()}}
            if not (dest / 'review.json').exists():
                atomic_write(dest / 'review.json', json.dumps(record, indent=2).encode())
            records.append(record)
        except (ValueError, TypeError, AttributeError, OSError, SyntaxError) as e:
            errors.append({'origin': origin, 'error': str(e)})
    # Include earlier snapshots even when an agent revises or removes its current submission.
    snapshots = [json.loads(p.read_text()) for p in sorted(out.glob('*/*/review.json'))]
    index = {'schema_version': 1, 'snapshots': snapshots, 'current_submissions': records, 'errors': errors}
    atomic_write(out / 'index.json', json.dumps(index, indent=2).encode())
    lines = ['# Detector candidates', '', 'Unreviewed agent submissions. Code and tests have not been executed.', '']
    for r in snapshots:
        lines.append(f"- `{r['id']}` — snapshot `{r['sha256'][:12]}`; source `{r['origin']}`; validation claim: {r['validation_claim']}")
    if errors:
        lines += ['', '## Invalid submissions', ''] + [f"- `{r['origin']}`: {r['error']}" for r in errors]
    atomic_write(out / 'index.md', ('\n'.join(lines) + '\n').encode())
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('traces', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args(argv)
    report = collect_candidates(a.traces, a.out)
    print(f"{len(report['snapshots'])} review snapshots; {len(report['errors'])} invalid submissions")
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
