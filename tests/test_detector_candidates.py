import json
from pathlib import Path

from lab.detector_candidates import collect_candidates, main


def candidate(root: Path, sid='s-1', id_='generic-method') -> Path:
    p = root / sid / 'detector_candidates' / id_
    p.mkdir(parents=True)
    (p / 'manifest.json').write_text(json.dumps({
        'schema_version': 1, 'id': id_, 'summary': 'Generic method', 'method': 'Compare statistics',
        'generalization': 'No fixed token/layer', 'requires_base': False, 'known_limitations': [],
        'provenance': {'source_suspect': sid, 'reference': 'trace step 12'}}))
    (p / 'detector.py').write_text(f"raise RuntimeError('must not execute')\nclass Candidate(Detector):\n    id = '{id_}'\n    default = False\n")
    (p / 'test_detector.py').write_text("raise RuntimeError('must not execute tests')\n")
    (p / 'README.md').write_text('Method and limitations')
    (p / 'validation.json').write_text(json.dumps({'status': 'untested', 'tests_run': [], 'resources': {}}))
    return p


def test_collect_is_nonexecuting_repeatable_and_preserves_revisions(tmp_path):
    root, out = tmp_path / 'traces', tmp_path / 'review'
    p = candidate(root)
    first = collect_candidates(root, out)
    assert len(first['snapshots']) == 1 and not first['errors']
    assert first['snapshots'][0]['review_status'] == 'unreviewed'
    assert collect_candidates(root, out) == first
    (p / 'README.md').write_text('Revised method')
    second = collect_candidates(root, out)
    assert len(second['snapshots']) == 2
    assert first['snapshots'][0] in second['snapshots']
    assert (p / 'README.md').read_text() == 'Revised method'


def test_invalid_submission_does_not_hide_valid_candidates(tmp_path):
    root, out = tmp_path / 'traces', tmp_path / 'review'
    candidate(root)
    bad = candidate(root, sid='s-2')
    (bad / 'detector.py').write_text('class Bad(Detector):\n    default = True\n')
    index = collect_candidates(root, out)
    assert len(index['snapshots']) == 1 and len(index['errors']) == 1
    assert main([str(root), '--out', str(out)]) == 1


def test_symlink_escape_and_false_provenance_rejected(tmp_path):
    root, out = tmp_path / 'traces', tmp_path / 'review'
    p = candidate(root)
    outside = tmp_path / 'outside.py'
    outside.write_text('sensitive')
    (p / 'detector.py').unlink()
    (p / 'detector.py').symlink_to(outside)
    assert collect_candidates(root, out)['errors']
    assert not list(out.glob('*/*/detector.py'))
    (p / 'detector.py').unlink()
    (p / 'detector.py').write_text("class Candidate(Detector):\n    id='generic-method'\n    default=False\n")
    manifest = json.loads((p / 'manifest.json').read_text())
    manifest['provenance']['source_suspect'] = 'different-suspect'
    (p / 'manifest.json').write_text(json.dumps(manifest))
    assert 'source_suspect' in collect_candidates(root, out)['errors'][0]['error']


def test_output_inside_traces_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError, match='outside traces'):
        collect_candidates(tmp_path, tmp_path / 'review')


def test_runner_collect_automatically_collates_candidates(tmp_path):
    import shutil
    from types import SimpleNamespace
    from lab.cleanroom.runner import HostRun, Options

    staged = tmp_path / 'staged'
    candidate(staged)
    work = tmp_path / 'work'
    work.mkdir()

    class Host:
        def run(self, command): return SimpleNamespace(out='yes')
        def get(self, remote, dst): shutil.copytree(staged / 's-1', dst, dirs_exist_ok=True)

    run = HostRun(spec='root@host', ids=['s-1'], workdir=work, key={}, repo='', ref='', keys={},
                  opts=Options(), host=Host())
    run.collect('s-1')
    index = json.loads((work / 'detector-review/index.json').read_text())
    assert len(index['snapshots']) == 1
    assert (work / 'traces/s-1/detector_candidates/generic-method/detector.py').exists()
