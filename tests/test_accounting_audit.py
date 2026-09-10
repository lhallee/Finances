"""Independent regressions for unfunded expenses and restricted grant cash."""

import numpy as np
import pytest

from finance_sim.business import BUSINESS_COLUMNS, funded_cost_ratio, simulate_business
from finance_sim.configuration import Grant, RunConfig
from finance_sim.engine import METRICS
from finance_sim.retirement import project_retirement
from finance_sim.scenarios import Scenario
from finance_sim.schedule import build_schedule


def empty_household():
    config = RunConfig(paths=1)
    config.household.medical_annual_oop = 0
    config.retirement.real_return = 0
    config.retirement.annual_volatility = 0
    config.retirement.annual_healthcare = 0
    config.retirement.annual_spending_floor = 10_000
    config.retirement.college_annual_cost = 0
    values = np.zeros((1, 123, len(METRICS)))
    values[:, :, METRICS.index('inflation_index')] = 1
    return config, values


def test_unpaid_child_expenses_do_not_create_retirement_income():
    config, values = empty_household()
    config.household.child_monthly_other = 10_000
    values[:, 3:, METRICS.index('child_other')] = 10_000
    unpaid = np.r_[np.zeros(3), np.arange(1, 121) * 10_000]
    values[:, :, METRICS.index('unpaid_bills')] = unpaid
    values[:, :, METRICS.index('liquid_wealth')] = -unpaid
    result = project_retirement(values, config, Scenario(births=(2027,), stop_after_birth=1))
    assert result.minimum_retirement_age.isna().all()
    assert result.projected_annual_saving_real.iloc[0] == -120_000
    assert result.projected_annual_saving_after_obligations_real.iloc[0] == 0


def test_tax_accrual_and_immediate_payment_infer_same_resources():
    config, accrued = empty_household()
    liability = np.arange(1, 124) * 100.
    accrued[:, :, METRICS.index('cash')] = 100_000
    accrued[:, :, METRICS.index('tax_payable')] = liability
    accrued[:, :, METRICS.index('liquid_wealth')] = 100_000 - liability
    paid = accrued.copy()
    paid[:, :, METRICS.index('tax_payable')] = 0
    paid[:, :, METRICS.index('cash')] = 100_000 - liability
    for household in (accrued, paid):
        result = project_retirement(household, config)
        assert result.projected_annual_saving_real.iloc[0] == -1200
        assert result.projected_annual_saving_after_obligations_real.iloc[0] == -1200
    assert project_retirement(accrued, config).equals(project_retirement(paid, config))


def test_advance_cannot_fund_another_projects_reimbursed_expenses():
    config = RunConfig(paths=1)
    config.business.cash = 0
    config.business.monthly_overhead = 0
    config.business.grants = [
        Grant(name='advance', amount=1200, start='2027-01-01', months=12,
              direct_cost_monthly=100, effort_fraction=.1, overhead_rate=0, milestones=((0, 1.),)),
        Grant(name='reimburse', amount=1200, start='2027-01-01', months=12,
              direct_cost_monthly=100, effort_fraction=.1, overhead_rate=0),
    ]
    scenario = Scenario(career='ud', company_outcome='dormant', grants=('advance', 'reimburse'))
    schedule = build_schedule(config, scenario)
    draws = np.zeros((1, len(schedule), 8))
    draws[:, :, 7] = .9
    company = simulate_business(config, scenario, schedule, draws)
    value = lambda name: company[0, 3, BUSINESS_COLUMNS.index(name)]
    assert value('operating_cost') == 200
    assert value('grant_receipts') == 200
    assert value('company_cash') == 0
    assert value('restricted_cash') == 1100


def test_partial_advance_never_invents_unrestricted_working_capital():
    # Half of cost is eligible for a project with only $5 of restricted cash.
    # No unrestricted cash means none of the other half can be paid.
    claims = np.array([10.])
    restricted = np.array([5.])
    assert funded_cost_ratio(20., 0., 0., claims, restricted) == pytest.approx(0.)
    # $5 of unrestricted cash can fund the uncovered half of $10 total spending.
    ratio = funded_cost_ratio(20., 5., 0., claims, restricted)
    assert ratio == pytest.approx(.5)
    assert 20 * ratio <= 5 + min(5, 10 * ratio) + 1e-10
