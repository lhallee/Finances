"""Read-only integrity and coverage checks for household or salary-support runs."""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from pathlib import Path


def checked_file(folder: Path, relative: str, expected: str) -> Path:
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError(f'Partition is outside the output folder: {relative}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'Partition hash mismatch: {relative}')
    return path


def validate_run(folder: Path, deep: bool = False) -> dict:
    """Check every committed partition; deep mode also checks row-level invariants.

    These checks establish saved-data consistency, not economic calibration or
    tax accuracy. A partial run can be valid but is never labeled complete.
    """
    salary = (folder / 'salary_support.json').exists()
    manifest = json.loads((folder / ('salary_support.json' if salary else 'manifest.json')).read_text(encoding='utf-8'))
    frames = []
    for part in manifest['parts']:
        if salary:
            paths = {kind: checked_file(folder, part[kind], part[f'{kind}_hash']) for kind in ('paths', 'cases')}
            cases = pd.read_parquet(paths['cases'], columns=['scenario_id', 'salary_2026'])
            frames.append(cases)
            if deep:
                retained = pd.read_parquet(paths['paths'])
                keys = ['dimension', 'choice', 'salary_2026', 'path_id']
                if retained.duplicated(keys).any():
                    raise ValueError('Duplicated paired paths inflate the effective sample size')
                counts = retained.groupby(keys[:-1]).size()
                if not counts.eq(manifest['paths_per_configuration']).all():
                    raise ValueError('Incomplete paired-path coverage')
                if len(cases) != manifest['expected_configurations']:
                    raise ValueError('Incomplete life-choice coverage at a salary point')
        else:
            paths = {kind: checked_file(folder, f'{kind}/{part["file"]}', part['hashes'][kind])
                     for kind in ('summary', 'terminal', 'monthly')}
            summary = pd.read_parquet(paths['summary'])
            if len(summary) != part['scenarios']:
                raise ValueError('Partition scenario count differs from its manifest')
            frames.append(summary[['scenario_id', 'paths']])
            if deep:
                terminal = pd.read_parquet(paths['terminal'])
                if terminal.duplicated(['scenario_id', 'path_id']).any():
                    raise ValueError('Duplicated terminal paths')
                counts = terminal.groupby('scenario_id').size()
                expected = summary.set_index('scenario_id').paths
                if not counts.reindex(expected.index).eq(expected).all() or set(counts.index) != set(expected.index):
                    raise ValueError('Terminal path counts differ from scenario summaries')
                medians = terminal.groupby('scenario_id').real_net_worth.median()
                if not np.allclose(medians.reindex(summary.scenario_id), summary.median_net_worth, rtol=1e-10, atol=1e-5):
                    raise ValueError('Net-worth summary does not reconcile to retained paths')
                bands = pd.read_parquet(paths['monthly'])
                if bands.duplicated(['scenario_id', 'date', 'metric']).any():
                    raise ValueError('Duplicated monthly bands')
                quantiles = bands[['q025', 'q100', 'q250', 'q500', 'q750', 'q900', 'q975']].to_numpy()  # (rows, 7)
                if not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < -1e-7).any():
                    raise ValueError('Monthly predictive ranges are nonfinite or unordered')
                config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
                months = 12 * (config['end_year'] - config['start_year'] + 1)
                if not bands.groupby(['scenario_id', 'metric']).size().eq(months).all():
                    raise ValueError('Monthly trajectory coverage is incomplete')
                if set(bands.scenario_id) != set(summary.scenario_id):
                    raise ValueError('Monthly and summary scenarios differ')
    columns = ['scenario_id', 'salary_2026'] if salary else ['scenario_id']
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    if rows.duplicated(columns).any():
        raise ValueError('Duplicate scenarios across committed partitions')
    completed = manifest['completed_salary_configurations' if salary else 'completed_scenarios']
    expected = manifest['expected_salary_configurations' if salary else 'expected_scenarios']
    if len(rows) != completed or completed > expected:
        raise ValueError('Manifest coverage does not reconcile to retained rows')
    if manifest['status'] == 'complete' and completed != expected:
        raise ValueError('Run is labeled complete but coverage is incomplete')
    return {'folder': str(folder.resolve()), 'kind': 'salary_support' if salary else 'household',
            'integrity': 'passed', 'deep': deep, 'status': manifest['status'],
            'completed': completed, 'expected': expected, 'partitions': len(manifest['parts']),
            'retirement_budget_model': manifest.get('retirement_budget_model', 'legacy'),
            'scope': 'Stored hashes, coverage and optional row invariants; not validation of model assumptions'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deep', action='store_true', help='Also reconcile paths, summaries and monthly bands')
    args = parser.parse_args()
    try:
        report = validate_run(args.output, args.deep)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Validation failed: {error}\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
