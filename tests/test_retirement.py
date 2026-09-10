"""Independent retirement capital checks, accessibility, censoring and projections."""

import numpy as np

from finance_sim.configuration import RunConfig, config_dict, config_from_dict
from finance_sim.engine import METRICS
from finance_sim.retirement import capital_factors, project_retirement
from finance_sim.results import SimulationResult, summarize
from finance_sim.business import BUSINESS_COLUMNS
from finance_sim.scenarios import Scenario


def example(cash=0, retirement=0):
    config = RunConfig(paths=2)
    config.retirement.annual_spending_override = 50_000
    config.retirement.real_return = 0
    config.retirement.annual_volatility = 0
    config.retirement.retirement_paths = 16
    values = np.zeros((2, 123, len(METRICS)))  # (paths, months, metrics)
    for key, value in (("cash", cash), ("liquid_wealth", cash), ("retirement", retirement), ("inflation_index", 1)):
        values[:, :, METRICS.index(key)] = value
    return config, values


def test_zero_return_required_capital_is_years_of_spending():
    np.testing.assert_allclose(capital_factors(0, 16, 72, 0, 0, .95), np.arange(73))
    config, values = example(cash=3_600_000)
    result = project_retirement(values, config)
    assert (result.minimum_retirement_age == 28.25).all()
    assert (result.retirement_required_capital_real == 72 * 50_000).all()
    assert (result.retirement_age_source == "within_simulation").all()


def test_locked_assets_wait_for_access_and_preserve_tax_haircut():
    config, values = example(retirement=5_000_000)
    result = project_retirement(values, config)
    assert (result.minimum_retirement_age == 60.25).all()
    assert (result.retirement_age_source == "beyond_horizon_projection").all()
    config.retirement.project_beyond_horizon = False
    assert project_retirement(values, config).minimum_retirement_age.isna().all()


def test_illiquid_value_cannot_fund_retirement_and_unreached_is_not_dropped():
    config, values = example()
    values[:, :, METRICS.index("business_interest")] = 1e9
    values[:, :, METRICS.index("home_equity")] = 1e9
    result = project_retirement(values, config)
    assert result.minimum_retirement_age.isna().all()
    summary, terminal, bands = summarize(SimulationResult(values, np.zeros((2, 123, len(BUSINESS_COLUMNS)))), config, Scenario())
    assert summary.retirement_reached_probability.iloc[0] == 0
    assert summary.median_minimum_retirement_age.isna().all()


def test_higher_budget_delays_readiness_and_config_roundtrip():
    config, values = example(cash=3_600_000)
    config.retirement.annual_spending_override = 100_000
    assert (project_retirement(values, config).minimum_retirement_age > 28.25).all()
    assert config_from_dict(config_dict(config)).retirement == config.retirement


def test_projection_reproducible_and_does_not_modify_ledger():
    config, values = example(cash=1_000_000)
    config.retirement.real_return = .03
    config.retirement.annual_volatility = .15
    copy = values.copy()
    a = project_retirement(values, config)
    assert a.equals(project_retirement(values, config))
    np.testing.assert_array_equal(copy, values)
