"""Decision rules must not manufacture certainty from repeated scenario rows."""

import dataclasses
import json

import numba
import numpy as np
import pandas as pd

from finance_sim.compensation import starting_salaries
from finance_sim.configuration import RunConfig, config_dict, config_from_dict
from finance_sim.decisions import frontier_cells, generate_decision_report, path_diagnostics
from finance_sim.engine import METRICS, simulate
from finance_sim.business import BUSINESS_COLUMNS
from finance_sim.results import SimulationResult, summarize
from finance_sim.scenarios import Scenario
from finance_sim.schedule import build_schedule, SCHEDULE


def test_salary_distribution_and_roundtrip():
    config = RunConfig(paths=100_000)
    bio = starting_salaries(config, "bio_ai_engineer")
    pharma = starting_salaries(config, "pharma_agritech_pi")
    assert bio.min() >= 200_000 and bio.max() <= 600_000
    assert (bio < 400_000).mean() > .65
    assert pharma.min() >= 100_000 and pharma.max() <= 500_000
    assert abs(np.median(pharma) - 300_000) < 1000
    np.testing.assert_array_equal(bio, starting_salaries(config, "protein_design"))
    config = config_from_dict(config_dict(config))
    np.testing.assert_array_equal(bio, starting_salaries(config, "bio_ai_engineer"))


def test_annual_raises_and_real_entry_salary():
    numba.set_num_threads(2)
    config = RunConfig(paths=2)
    config.household.job_loss_annual_probability = 0
    draws = np.zeros((2, 123, 8))
    draws[:, :, 1] = .002
    draws[:, :, 4] = .003
    draws[:, :, 5:8] = .9
    scenario = Scenario(career="bio_ai_engineer", career_year=2027, company_outcome="dormant", grants=())
    result = SimulationResult(*simulate(config, scenario, draws))
    salaries = starting_salaries(config, scenario.career)
    np.testing.assert_allclose(result.series("logan_income")[:, 3] * 12 / result.series("inflation_index")[:, 3], salaries)
    np.testing.assert_allclose(result.series("logan_income")[:, 3:15], np.repeat(result.series("logan_income")[:, 3:4], 12, axis=1))
    np.testing.assert_allclose(result.series("logan_income")[:, 15] / result.series("logan_income")[:, 14], 1.003 ** 12)
    schedule = build_schedule(config, scenario)
    assert schedule[15, SCHEDULE.logan_raise_due] == 1


def test_ud_remains_employed_even_when_job_draws_trigger_other_roles():
    config = RunConfig(paths=2)
    config.household.job_loss_annual_probability = 1
    draws = np.zeros((2, 123, 8))
    result = SimulationResult(*simulate(config, Scenario(career="ud", company_outcome="dormant", grants=()), draws))
    assert (result.series("logan_income")[:, 3:] > 0).all()


def test_depletion_is_not_negative_initial_net_worth_and_tracks_reversal():
    config = RunConfig(paths=2)
    household = np.zeros((2, 123, len(METRICS)))
    household[:, :, METRICS.index("inflation_index")] = 1
    household[:, :, METRICS.index("net_worth")] = -100
    household[:, :, METRICS.index("liquid_wealth")] = 100
    household[:, :, METRICS.index("cash")] = 100
    household[1, 8:10, METRICS.index("net_worth")] = 100
    household[1, 10:, METRICS.index("liquid_wealth")] = -10
    result = SimulationResult(household, np.zeros((2, 123, len(BUSINESS_COLUMNS))))
    rows = path_diagnostics(result, config, Scenario(career_year=2029))
    assert rows.ever_liquid_depleted.tolist() == [False, True]
    assert rows.positive_then_negative_net_worth.tolist() == [False, True]
    assert rows.depletion_before_new_role.tolist() == [False, True]
    assert pd.isna(rows.first_liquid_depletion.iloc[0])
    assert rows.first_liquid_depletion.iloc[1] == pd.Timestamp("2027-08-01")
    # Quantiles must deflate each path, not divide two independent medians.
    household[0, :, METRICS.index("inflation_index")] = 2
    _, _, monthly = summarize(result, config, Scenario())
    value = monthly[monthly.metric == "net_worth"].iloc[0].q500
    assert value == -75


def test_empty_salary_tail_prevents_unsupported_threshold():
    config = RunConfig()
    rows = pd.DataFrame({"salary_family": "bio_ai", "macro": "baseline", "path_id": np.arange(1000),
                         "starting_salary_2026": 225_000., "ever_distress": False,
                         "depletion_before_new_role": False, "tested_scenarios": 10})
    cells = frontier_cells(rows, config)
    assert cells.meets_risk_budget.iloc[0]
    assert cells.supported_threshold_2026.isna().all()
    assert (cells[cells.independent_paths == 0].risk_upper_simultaneous_95 == 1).all()


def test_repeated_scenarios_do_not_increase_independent_sample_size(tmp_path):
    config = RunConfig(paths=100)
    (tmp_path / "terminal").mkdir()
    (tmp_path / "config.json").write_text(json.dumps(config_dict(config)))
    frames = []
    for scenario in ("a", "b"):
        frames.append(pd.DataFrame({"scenario_id": scenario, "path_id": np.arange(100),
                                    "starting_salary_2026": 225_000., "ever_distress": np.arange(100) < 10,
                                    "depletion_before_new_role": False}))
    pd.concat(frames).to_parquet(tmp_path / "terminal" / "part.parquet")
    (tmp_path / "manifest.json").write_text(json.dumps({"parts": [{"file": "part.parquet"}], "completed_scenarios": 2, "expected_scenarios": 2}))
    summary = pd.DataFrame({"scenario_id": ["a", "b"], "career": "bio_ai_engineer", "macro": "baseline", "distress_probability": .1})
    _, cells = generate_decision_report(tmp_path, summary)
    observed = cells[(cells.macro == "baseline") & (cells.independent_paths > 0)].iloc[0]
    assert observed.independent_paths == 100
    assert observed.distress_paths == 10
    assert observed.risk_estimate == .1
    assert observed.tested_scenarios == 2
