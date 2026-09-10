"""Independent arithmetic and event tests for financially material behavior."""

import dataclasses

import numba
import numpy as np
import pytest

from finance_sim.configuration import FundingRound, Grant, Loan, RunConfig, config_dict, config_from_dict
from finance_sim.economics import economic_paths
from finance_sim.engine import METRICS, payment, simulate
from finance_sim.equity import dilute, exit_waterfall, qsbs_exclusion
from finance_sim.results import SimulationResult, probability_interval
from finance_sim.scenarios import Scenario, count_scenarios, generate_scenarios
from finance_sim.schedule import SCHEDULE, build_schedule
from finance_sim.taxes import annual_tax, federal_income_tax, payroll_tax, state_income_tax


numba.set_num_threads(2)


@pytest.fixture
def config():
    cfg = RunConfig(paths=4, generate_reports=False)
    cfg.household.job_loss_annual_probability = 0
    cfg.household.repair_annual_probability = 0
    return cfg


def result(config, scenario):
    return SimulationResult(*simulate(config, scenario))


def test_federal_single_brackets():
    # $16,100 standard deduction + $12,400 at 10% + $27,600 at 12%.
    assert federal_income_tax(56_100, 0., 0., False, 0, 56_100, 0., 0., 1.) == pytest.approx(4552)


def test_joint_brackets():
    assert federal_income_tax(112_200, 0., 0., True, 0, 56_100, 56_100, 0., 1.) == pytest.approx(9104)


def test_payroll_person_cap_and_self_employment():
    assert payroll_tax(200_000, 0, 184_500) == pytest.approx(184_500*.062 + 200_000*.0145)
    assert payroll_tax(0, 300, 184_500) == 0
    assert payroll_tax(0, 10_000, 184_500) == pytest.approx(10_000*.9235*.153)


@pytest.mark.parametrize("state", ["PA", "DE", "ME", "NY", "CA", "MA"])
def test_state_names_and_monotonicity(state):
    assert state_income_tax(0, state, False, 1.) == 0
    assert state_income_tax(200_000, state, False, 1.) > state_income_tax(100_000, state, False, 1.)


def test_mortgage_schedule():
    monthly = payment(300_000, .06/12, 360)
    assert monthly == pytest.approx(1798.651575)
    balance = 300_000
    for _ in range(360):
        balance = balance*1.005-monthly
    assert abs(balance) < .001
    assert payment(1200, 0, 12) == 100


def test_dilution():
    share, existing, new = dilute(.315, np.zeros(1), 10e6, 30e6, .1)
    assert share == pytest.approx(.315*.9*.75)
    assert new == .25


def test_preference_small_exit():
    founder, advisor, investors, total = exit_waterfall(5e6, 0., 0., 0., .75,
        np.array([.25]), np.array([10e6]), np.array([1.]), np.array([0.]), 0.)
    assert founder == 0
    assert investors[0] == 5e6
    assert total == investors.sum()


def test_preference_conversion_large_exit():
    founder, advisor, investors, total = exit_waterfall(100e6, 0., 0., 0., .75,
        np.array([.25]), np.array([10e6]), np.array([1.]), np.array([0.]), 0.)
    assert founder == 75e6
    assert investors[0] == 25e6
    assert founder + advisor + investors.sum() == total


def test_participating_preference():
    founder, advisor, investors, total = exit_waterfall(100e6, 0., 0., 0., .75,
        np.array([.25]), np.array([10e6]), np.array([1.]), np.array([1.]), 0.)
    assert investors[0] == 10e6 + 90e6*.25
    assert founder == 90e6*.75


def test_advisor_not_duplicate_equity():
    founder, advisor, investors, total = exit_waterfall(100e6, 0., 0., 0., .35*.9*.999,
        np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), .005)
    assert advisor == 500_000
    assert founder == pytest.approx(99.5e6*.35*.9*.999)


def test_qsbs_dates_and_cap():
    assert qsbs_exclusion(20e6, 0, 0, 35, True) == 0
    assert qsbs_exclusion(20e6, 0, 0, 36, True) == 7.5e6
    assert qsbs_exclusion(20e6, 0, 0, 48, True) == 11.25e6
    assert qsbs_exclusion(20e6, 0, 0, 60, True) == 15e6
    assert qsbs_exclusion(20e6, 0, 0, 60, False) == 0


@pytest.mark.parametrize("stop", [1, 2, 3])
def test_amanda_stops_at_selected_birth(config, stop):
    scenario = Scenario(births=(2028, 2030, 2032), stop_after_birth=stop, marriage_year=2027)
    r = result(config, scenario)
    event = (scenario.births[stop-1]-2026)*12-9
    assert np.all(r.series("amanda_income")[:, event:] == 0)
    assert np.all(r.series("childcare")[:, event:] == 0)
    assert np.all(r.series("amanda_employee_retirement")[:, event:] == 0)
    assert np.all(r.series("amanda_employer_retirement")[:, event:] == 0)
    assert np.all(r.series("child_other")[:, event:] > 0)
    assert np.all(r.series("amanda_retirement")[:, event:] > 0)
    assert np.all(r.series("amanda_income")[:, event-1] > 0)


def test_missing_birth_does_not_stop_work(config):
    r = result(config, Scenario(births=(2029,), stop_after_birth=3))
    assert np.all(r.series("amanda_income") > 0)


def test_unmarried_health_eligibility(config):
    scenario = Scenario(births=(2027,), stop_after_birth=1, career="ud")
    ineligible = result(config, scenario)
    config.household.domestic_partner_eligible = True
    eligible = result(config, scenario)
    assert np.all(ineligible.series("health_premium")[:, 3:] > eligible.series("health_premium")[:, 3:])


def test_blend_start_and_benefits(config):
    schedule = build_schedule(config, Scenario(benefits="lost"))
    assert schedule[3, SCHEDULE.logan_wage] == 70_000/12
    assert schedule[4, SCHEDULE.logan_wage] == 31_500/12
    assert schedule[4, SCHEDULE.founder_fte] == .55
    assert schedule[4, SCHEDULE.logan_benefits] == 0


def test_no_grant_is_not_personal_cash(config):
    r = result(config, Scenario(grants=(), company_outcome="dormant", career="ud"))
    assert np.all(r.series("grant_receipts") == 0)
    assert np.all(r.series("company_pay") == 0)


def test_delayed_grant_records_receivables(config):
    r = result(config, Scenario(grant_case="delayed"))
    assert np.all(r.series("grant_receipts")[:, 4:7] == 0)
    assert np.any(r.series("company_receivables")[:, 4:7] > 0)
    assert np.all(r.series("company_cash") >= 0)


def test_home_purchase_and_reconciliation(config):
    config.household.logan.cash = 800_000
    config.household.amanda.cash = 200_000
    scenario = Scenario(career="profluent", location="squirrel_hill", housing="buy", move_year=2027, marriage_year=2027)
    r = result(config, scenario)
    assert np.any(r.series("purchased") == 1)
    assert np.max(np.abs(r.series("reconciliation_error"))) < .01


def test_zero_income_no_invented_credit(config):
    config.household.logan.salary = 0
    config.household.amanda.salary = 0
    config.household.logan.cash = 0
    config.household.amanda.cash = 0
    config.household.logan.brokerage = 0
    config.household.amanda.brokerage = 0
    r = result(config, Scenario(career="ud", company_outcome="dormant", grants=()))
    assert np.any(r.series("shortfall") > 0)
    assert np.all(r.series("cash") >= 0)
    assert np.isfinite(r.household).all()


def test_tax_settles_in_april(config):
    r = result(config, Scenario())
    months = np.array([(10+m-1)%12+1 for m in range(r.household.shape[1])])
    assert np.all(r.series("tax_settlement")[:, months != 4] == 0)


def test_exit_cash_and_taxes(config):
    config.business.rounds = [FundingRound(2028, 1e6, 4e6, 0)]
    r = result(config, Scenario(company_outcome="vc", exit_value=10e6, exit_year=2030, exit_type="equity"))
    assert np.any(r.series("exit_cash") > 0)
    assert np.any(r.series("tax_expense") > 10_000)
    assert np.max(np.abs(r.series("reconciliation_error"))) < .01


def test_small_grid_count_matches_enumeration(config):
    grid = config.grid
    grid.years = (2027, 2028)
    grid.careers = ("ud", "profluent")
    grid.locations = ("parents", "bay_area")
    grid.amanda_careers = ("pharma",)
    grid.company_modes = ("business",)
    grid.company_outcomes = ("dormant", "saas")
    grid.macros = ("baseline",)
    grid.exit_values = (0, 10e6)
    grid.funding_portfolios = ((), ("darpa",))
    rows = list(generate_scenarios(config, "full"))
    assert len(rows) == count_scenarios(config, "full")
    assert len({s.id for s in rows}) == len(rows)


def test_config_roundtrip(config):
    restored = config_from_dict(config_dict(config))
    assert config_dict(restored) == config_dict(config)


def test_reproducible_paths(config):
    first = result(config, Scenario())
    second = result(config, Scenario())
    np.testing.assert_array_equal(first.household, second.household)


def test_interval_boundary():
    low, high = probability_interval(0, 256)
    assert low == 0 and 0 < high < .02


def test_practical_schema_coverage(config):
    config.grid.independent_career_locations = False
    config.grid.base_schemas = {"base": {}}
    config.grid.practical_birth_schedules = ((), (2029,), (2029, 2031))
    config.grid.practical_marriage_years = (0, 2028)
    config.grid.macros = ("baseline", "recession")
    rows = list(generate_scenarios(config, "standard"))
    assert len(rows) == (1 + 2 + 3) * 2 * 2 * 2
    assert len({s.id for s in rows}) == count_scenarios(config, "standard")


def test_grant_claims_cannot_exceed_spending(config):
    config.business.grants[0].overhead_rate = 10
    r = result(config, Scenario())
    assert np.all(r.series("grant_claims") <= r.series("operating_cost") + .001)


def test_exit_basis_recovered_once_and_employment_continues(config):
    config.business.equity_basis = 100_000
    config.business.exit_escrow_fraction = .2
    r = result(config, Scenario(career="synthyra", exit_value=10e6, exit_year=2030))
    np.testing.assert_allclose(r.series("exit_basis_recovered").sum(axis=1), 100_000)
    assert np.all(r.series("company_pay")[:, 52:] > 0)
    assert np.max(np.abs(r.series("reconciliation_error"))) < .01


def test_forecast_award_failure_retains_ud(config):
    config.business.grants[0].award_probability = 0
    r = SimulationResult(*simulate(config, Scenario(), forecast=True))
    assert np.all(r.series("darpa_awarded") == 0)
    assert np.all(r.series("company_pay") == 0)
    assert np.all(r.series("logan_income")[:, 4] >= config.household.logan.salary / 12 * .9)


def test_dependent_care_credit_2026():
    without = federal_income_tax(100_000, 0., 0., True, 2, 50_000, 50_000, 0., 1.)
    with_care = federal_income_tax(100_000, 0., 0., True, 2, 50_000, 50_000, 6000., 1.)
    assert without - with_care == pytest.approx(2100)
    stay_home = federal_income_tax(100_000, 0., 0., True, 2, 100_000, 0., 6000., 1.)
    assert without == stay_home


def test_salt_expiration_and_editable_tax_anchors():
    args = (150_000., 100_000., 0., 0., 0., 0., 0., 0., 0., True, 0, 0., "CA", 0., 0., 1.)
    before = annual_tax(*args, property_tax=20_000, tax_year=2029)
    after = annual_tax(*args, property_tax=20_000, tax_year=2030)
    assert after[0] > before[0]
    assert annual_tax(*args, ss_base=50_000)[2] < annual_tax(*args)[2]
    assert annual_tax(*args, standard_single=50_000)[0] < annual_tax(*args)[0]


def test_deferred_loan_interest_is_simple(config):
    config.household.loans = [Loan("Deferred", 1000, .12, 0, "2027-09-30")]
    config.household.cash_target_monthly = 1e9
    r = result(config, Scenario(career="ud", grants=()))
    np.testing.assert_allclose(r.series("debt_interest")[:, :12], 10.)
    np.testing.assert_allclose(r.series("debt")[:, 11], 1120.)
    assert np.all(r.series("debt_payment")[:, 12] > 0)


def test_retirement_saving_stops_before_unmet_bills(config):
    config.household.logan.cash = 0
    config.household.amanda.cash = 0
    config.household.logan.monthly_personal_spend = 100_000
    config.household.amanda.monthly_personal_spend = 100_000
    r = result(config, Scenario(career="ud", grants=()))
    assert np.all(r.series("employee_retirement") == 0)


def test_tax_components_reconcile_after_exit(config):
    r = result(config, Scenario(company_outcome="vc", exit_value=100e6, exit_year=2031))
    components = r.series("federal_tax") + r.series("state_tax") + r.series("payroll_tax")
    np.testing.assert_allclose(components[:, 3:], r.series("tax_expense")[:, 3:], atol=.01)
