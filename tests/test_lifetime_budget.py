"""Dated cost boundaries and independently computed loan cash payments."""
import numpy as np
import pytest

from finance_sim.configuration import RunConfig
from finance_sim.engine import METRICS
from finance_sim.lifetime_budget import lifetime_budget
from finance_sim.retirement import project_retirement
from finance_sim.scenarios import Scenario


def household():
    config = RunConfig(paths=2)
    values = np.zeros((2, 123, len(METRICS)))
    values[:, :, METRICS.index('inflation_index')] = 1.
    config.retirement.future_inflation = 0
    return config, values


def test_college_replaces_child_living_and_ends_at_22_with_partial_years():
    config, values = household()
    config.grid.event_month = 9
    config.retirement.college_annual_cost = 30000
    config.retirement.college_parent_share = .5
    case = Scenario(births=(2029,), stop_after_birth=1)
    budget = lifetime_budget(values, config, case)
    year = lambda y: y - 2027
    assert budget.child_support[0, year(2046)] == 5400
    assert budget.child_support[0, year(2047)] == 3600  # Jan through August.
    assert budget.college[0, year(2047)] == 5000  # September through December.
    assert budget.college[0, year(2050)] == 15000
    assert budget.college[0, year(2051)] == 10000
    assert np.all(budget.child_support[:, year(2048):] == 0)
    assert np.all(budget.college[:, year(2052):] == 0)
    assert np.all(budget.childcare == 0)


@pytest.mark.parametrize('term', [20, 25, 30])
def test_mortgage_ends_at_actual_term_and_owner_expenses_continue(term):
    config, values = household()
    config.housing.mortgage_years = term
    config.housing.mortgage_rate = .06
    principal, rate = 400000., .06 / 12
    monthly = principal * rate / (1 - (1 + rate) ** (-term * 12))
    paid = 119  # February 2027 through December 2036.
    balance = principal * (1 + rate) ** paid - monthly * ((1 + rate) ** paid - 1) / rate
    values[:, 3:, METRICS.index('home_value')] = 500000
    values[:, -1, METRICS.index('mortgage')] = balance
    values[:, -1, METRICS.index('property_tax')] = 500
    budget = lifetime_budget(values, config)
    end = 2027 + term
    assert budget.mortgage[0, end - 2027] == pytest.approx(monthly)
    assert np.all(budget.mortgage[:, end - 2027 + 1:] == 0)
    assert budget.mortgage[0, 10:].sum() == pytest.approx((term * 12 - paid) * monthly)
    assert np.all(budget.housing_ongoing[:, end - 2027:] > 6000)


def test_rent_continues_and_fixed_mortgage_deflates():
    config, values = household()
    values[:, :, METRICS.index('housing_cost')] = 3000
    budget = lifetime_budget(values, config)
    assert np.all(budget.housing_ongoing[:, 10:] == 36000)
    assert np.all(budget.mortgage == 0)
    values[:, 3:, METRICS.index('home_value')] = 500000
    values[:, -1, METRICS.index('mortgage')] = 300000
    config.retirement.future_inflation = .03
    budget = lifetime_budget(values, config)
    assert budget.mortgage[0, 11] < budget.mortgage[0, 10]


def test_expiring_obligations_increase_future_saving_and_are_reserved():
    config, values = household()
    values[:, :, METRICS.index('cash')] = 100000
    values[:, :, METRICS.index('liquid_wealth')] = 100000
    values[:, :, METRICS.index('child_other')] = 5400 / 12
    values[:, :, METRICS.index('living_cost')] = 2000
    values[:, :, METRICS.index('medical_out_of_pocket')] = 150
    result = project_retirement(values, config, Scenario(births=(2029,), stop_after_birth=1))
    assert np.all(result.projected_annual_saving_after_obligations_real > result.projected_annual_saving_real + 5000)
    assert result.retirement_budget_model.eq('lifecycle_v2').all()


def test_future_college_cost_does_not_disappear_at_early_retirement():
    config, values = household()
    config.retirement.annual_healthcare = 0
    config.retirement.annual_spending_floor = 0
    budget = lifetime_budget(values, config, Scenario(births=(2029,)))
    assert budget.temporary_retirement.sum() >= 2 * 4 * config.retirement.college_annual_cost * .5
