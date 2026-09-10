"""Saved-run integrity must fail visibly when committed evidence changes."""

import pytest

from finance_sim.configuration import RunConfig
from finance_sim.workflow import execute
from validate_run import validate_run


def test_validator_reconciles_and_detects_corruption(tmp_path):
    manifest = execute(RunConfig(paths=2, threads=2, generate_reports=False), tmp_path, preset='baseline')
    report = validate_run(tmp_path, deep=True)
    assert report['integrity'] == 'passed' and report['completed'] == 1
    (tmp_path / 'monthly' / manifest['parts'][0]['file']).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_run(tmp_path)
