"""No-company-income support criteria and common-draw inference."""

import json

import numba
import numpy as np
import pandas as pd
import pytest

from dashboard_insights import risk_trends
from finance_sim.business import BUSINESS, BUSINESS_COLUMNS
from finance_sim.configuration import RunConfig, config_dict, config_from_dict
from finance_sim.engine import METRICS, simulate
from finance_sim.results import SimulationResult
from finance_sim.salary_support import support_scenarios, support_outcomes, support_statistics, run_salary_support


def test_scope_and_configuration_roundtrip():
    config = RunConfig()
    cases = support_scenarios(config)
    assert len(cases) == 1548
    assert {s.location for s in cases} == set(config.grid.locations)
    assert {s.macro for s in cases} == set(config.grid.macros)
    assert {len(s.births) for s in cases} == {1, 2, 3}
    assert all(0 < s.stop_after_birth <= len(s.births) and not s.grants and s.exit_value == 0 for s in cases)
    assert config_from_dict(config_dict(config)).salary_support.retirement_age == 60


def test_saved_birth_arrays_preserve_scenario_identifiers():
    from finance_sim.scenarios import scenario_from_record
    scenario = support_scenarios(RunConfig())[0]
    record = scenario.record()
    record['births'] = np.asarray(record['births'], dtype=np.int64)
    assert scenario_from_record(record).id == scenario.id


def test_darpa_only_pays_for_24_months_then_switches_without_overlap():
    from finance_sim.economics import monthly_dates
    from finance_sim.schedule import build_schedule, SCHEDULE
    config = RunConfig(paths=2, threads=2)
    config.salary_support.company_support = 'darpa_only'
    config.household.job_loss_annual_probability = 0
    config.business.grants[0].follow_on_probability = 0
    config.business.distributions_fraction = 0
    config.career_salary_overrides['bio_ai_engineer'] = 300000
    scenario = support_scenarios(config)[0]
    assert (scenario.career_year, scenario.career_month, scenario.prior_career) == (2029, 2, 'ud_blend')
    assert scenario.grants == ('darpa',) and scenario.company_outcome == 'dormant'
    schedule = build_schedule(config, scenario)
    dates = monthly_dates(config)
    funded = (dates >= '2027-02-01') & (dates < '2029-02-01')
    assert funded.sum() == 24
    assert np.all(schedule[funded, SCHEDULE.founder_fte] == .55)
    assert np.all(schedule[funded, SCHEDULE.ud_wage] == 31500 / 12)
    assert np.all(schedule[~funded, SCHEDULE.founder_fte] == 0)
    assert np.all(schedule[dates >= '2029-02-01', SCHEDULE.logan_wage] == 300000 / 12)
    result = SimulationResult(*simulate(config, scenario))
    assert np.all(result.business[:, funded, BUSINESS.founder_pay] > 0)
    assert np.all(result.business[:, ~funded, BUSINESS.founder_pay] == 0)
    assert np.all(result.business[:, :, BUSINESS.revenue] == 0)
    support_outcomes(result, config, scenario)
    result.business[0, -1, BUSINESS.founder_pay] = 1
    with pytest.raises(ArithmeticError, match='outside'):
        support_outcomes(result, config, scenario)


def test_all_synthyra_income_channels_are_zero_and_detected():
    numba.set_num_threads(2)
    config = RunConfig(paths=2)
    config.household.job_loss_annual_probability = 0
    config.career_salary_overrides['bio_ai_engineer'] = 400_000
    scenario = support_scenarios(config)[0]
    result = SimulationResult(*simulate(config, scenario))
    observed = support_outcomes(result, config, scenario)
    assert len(observed['retirement_age']) == 2
    for column in (BUSINESS.founder_pay, BUSINESS.distribution, BUSINESS.equity_proceeds, BUSINESS.ordinary_proceeds):
        modified = result.business.copy()
        modified[0, 0, column] = 1
        with pytest.raises(ArithmeticError, match='Synthyra'):
            support_outcomes(SimulationResult(result.household, modified), config, scenario)


def test_reserves_after_stop_and_required_debt_not_optional_extra():
    config = RunConfig(paths=2)
    values = np.zeros((2, 123, len(METRICS)))
    for field, value in {'inflation_index': 1, 'liquid_wealth': 7000, 'cash': 7000, 'living_cost': 1000,
                         'debt_payment': 10000, 'debt_extra': 10000}.items():
        values[:, :, METRICS.index(field)] = value
    values[:, 30:, METRICS.index('amanda_stopped')] = 1
    values[0, 3:20, METRICS.index('liquid_wealth')] = 100  # Low buffer before Amanda stops is not a reserve failure.
    values[1, 35, METRICS.index('liquid_wealth')] = 5000
    outcome = support_outcomes(SimulationResult(values, np.zeros((2, 123, len(BUSINESS_COLUMNS)))), config, support_scenarios(config)[0])
    assert outcome['reserve_failure'].tolist() == [False, True]
    assert not outcome['distress'].any()
    values[0, 35, METRICS.index('shortfall')] = 10
    outcome = support_outcomes(SimulationResult(values, np.zeros((2, 123, len(BUSINESS_COLUMNS)))), config, support_scenarios(config)[0])
    assert outcome['distress'].tolist() == [True, False]


def test_age_target_union_and_nonmonotonic_salary_thresholds():
    rows = []
    for salary, age in [(200000, 65.), (300000, 59.), (400000, 70.), (500000, 55.)]:
        rows.append(pd.DataFrame({'dimension': 'overall', 'choice': 'all', 'salary_2026': salary,
                                 'path_id': np.arange(256), 'tested_configurations': 1000,
                                 'distress': False, 'reserve_failure': False, 'pre_salary_distress': False,
                                 'retirement_age': age, 'minimum_liquid_2026': 100_000, 'minimum_reserve_margin_2026': 30_000}))
    paths = pd.concat(rows)
    stats = support_statistics(paths, retirement_age=60)
    assert stats.supported_salary_2026.eq(500000).all()  # Passing 300k does not hide failing 400k.
    assert stats.independent_paths.eq(256).all()  # Never 256,000 correlated scenario-path trials.
    assert support_statistics(paths, retirement_age=75).supported_salary_2026.eq(200000).all()
    assert support_statistics(paths, retirement_age=75, minimum_paths=512).supported_salary_2026.isna().all()
    paths.loc[paths.salary_2026 == 500000, 'reserve_failure'] = True
    assert support_statistics(paths, retirement_age=60).supported_salary_2026.isna().all()


def test_small_runner_preserves_inputs_and_completed_output(tmp_path):
    config = RunConfig(paths=2, threads=2)
    config.grid.locations = ('parents',)
    config.grid.macros = ('baseline',)
    config.grid.practical_birth_schedules = ((2029,),)
    config.salary_support.salaries = (200000, 400000)
    original = config_dict(config)
    folder = tmp_path / 'support'
    run_salary_support(config, folder, progress=lambda *a, **kw: None)
    assert config_dict(config) == original
    manifest = json.loads((folder / 'salary_support.json').read_text())
    assert manifest['status'] == 'complete' and manifest['completed_salary_configurations'] == 2
    with pytest.raises(FileExistsError):
        run_salary_support(config, folder)
    again = run_salary_support(config, folder, resume=True)
    assert again == manifest
    paths = pd.read_parquet(folder / manifest['parts'][0]['paths'])
    assert len(paths[paths.dimension == 'overall']) == 2


def test_risk_trends_are_configuration_averages_not_examples():
    frame = pd.DataFrame({'location': ['nyc', 'nyc', 'parents'], 'scenario_id': ['a', 'b', 'c'],
                          'distress_probability': [0., 1., .2], 'liquid_depletion_probability': [0., 1., .2],
                          'shortfall_probability': [0., .5, .1]})
    trend = risk_trends(frame, 'location').set_index('location')
    assert trend.loc['nyc', 'average_risk'] == .5
    assert trend.loc['nyc', 'configurations'] == 2
    assert trend.loc['parents', 'average_risk'] == .2


def test_salary_panel_recalculates_requirements_without_simulation(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    folder = tmp_path / 'salary'
    folder.mkdir()
    frames = []
    for salary, retirement in [(200000, 65.), (400000, 55.)]:
        for dimension, choice in [('overall', 'all'), ('location', 'parents')]:
            frames.append(pd.DataFrame({'dimension': dimension, 'choice': choice, 'salary_2026': salary,
                                        'path_id': np.arange(256), 'tested_configurations': 1, 'distress': False,
                                        'reserve_failure': False, 'pre_salary_distress': False, 'retirement_age': retirement,
                                        'minimum_liquid_2026': 100000., 'minimum_reserve_margin_2026': 50000.}))
    pd.concat(frames).to_parquet(folder / 'paths.parquet')
    metadata = {'status': 'complete', 'retirement_age': 60, 'reserve_months': 6, 'fingerprint': 'test',
                'risk_budget': .05, 'completed_salary_configurations': 2, 'expected_salary_configurations': 2,
                'expected_configurations': 1, 'salaries_2026': [200000, 400000], 'paths_per_configuration': 256,
                'parts': [{'paths': 'paths.parquet'}], 'retirement_funding_target': .95, 'retirement_life_expectancy': 100,
                'coverage': {'salary_start_year': 2027, 'marriage_year': 2028, 'move_years': [2029],
                             'birth_schedules': [[2029]], 'locations': ['parents'], 'macros': ['baseline']}, 'notice': 'Test scope'}
    manifest = folder / 'salary_support.json'
    manifest.write_text(json.dumps(metadata))
    monkeypatch.setattr('dashboard_insights.latest_support', lambda outputs: folder)
    app = AppTest.from_string("from pathlib import Path\nfrom dashboard_insights import show_salary_support\nshow_salary_support(Path('unused'))")
    app.run(timeout=60)
    assert not app.exception
    assert app.metric[0].value == '$400,000'
    next(box for box in app.selectbox if box.label == 'Retirement age').set_value(65).run(timeout=60)
    assert not app.exception
    assert app.metric[0].value == '$200,000'
    assert app.get('button_group')[0].value == ['retirement', 'bills', 'reserve']
    next(box for box in app.selectbox if box.label == 'Retirement age').set_value(60).run(timeout=60)
    assert app.metric[0].value == '$400,000'
    app.get('button_group')[0].set_value(['bills', 'reserve']).run(timeout=60)
    assert not app.exception
    assert app.metric[0].value == '$200,000'
    assert len(app.get('plotly_chart')) == 2
    app.get('button_group')[0].set_value([]).run(timeout=60)
    assert not app.exception
    assert not app.metric
    assert any('at least one' in item.value for item in app.info)
    app.get('button_group')[0].set_value(['retirement', 'bills', 'reserve']).run(timeout=60)
    assert app.metric[0].value == '$400,000'
    metadata['status'] = 'running'
    manifest.write_text(json.dumps(metadata))
    app.run(timeout=60)
    assert app.metric[0].value == 'Computing'
    assert any('withheld' in item.value for item in app.info)
    metadata['status'] = 'complete'
    metadata['fingerprint'] = 'noisy-test'
    noisy = pd.concat(frames)
    noisy.loc[noisy.path_id < 10, 'retirement_age'] = 79.
    noisy.to_parquet(folder / 'paths.parquet')
    manifest.write_text(json.dumps(metadata))
    app.run(timeout=60)
    assert not app.exception
    assert app.metric[0].value == 'More paths needed'
    assert any('sampling precision' in item.value for item in app.info)
