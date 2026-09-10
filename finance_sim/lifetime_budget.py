"""Finite child support and amortizing housing costs, all in 2026 dollars."""

from dataclasses import dataclass
from math import ceil

import numpy as np

from .configuration import RunConfig
from .engine import METRICS


@dataclass(frozen=True)
class LifetimeBudget:
    years: np.ndarray
    ongoing_retirement: np.ndarray  # (paths,): adult living, health, rent/owner expenses.
    temporary_retirement: np.ndarray  # (paths, years): fully reserved finite obligations.
    working_spending: np.ndarray  # (paths, years): recurring spending, without saving/tax.
    child_support: np.ndarray
    college: np.ndarray
    childcare: np.ndarray
    mortgage: np.ndarray
    housing_ongoing: np.ndarray
    nonrecurring: np.ndarray


def lifetime_budget(household: np.ndarray, config: RunConfig, scenario=None) -> LifetimeBudget:
    """Extend actual ledgers using dated obligations, not perpetual final-year costs.

    No new home purchase is invented beyond the saved horizon. Existing homes
    retain their space and real operating costs; renters retain their real rent.
    College replaces ordinary child living support, including room and board.
    """
    policy, home = config.retirement, config.housing
    paths, months = household.shape[:2]
    observed_years = (months - 3) // 12
    lifetime = ceil(policy.life_expectancy - min(config.household.logan.age, config.household.amanda.age))
    years = np.arange(config.start_year, config.start_year + lifetime)
    count = len(years)

    def series(name):
        return household[:, :, METRICS.index(name)]

    inflation = series('inflation_index')

    def annual(name):
        return (series(name)[:, 3:] / inflation[:, 3:]).reshape(paths, observed_years, 12).sum(axis=2)

    def last_annual(name):
        return series(name)[:, -1] / inflation[:, -1] * 12

    living = last_annual('living_cost')
    owner = series('home_value')[:, -1] > 0
    real_home = series('home_value')[:, -1] / inflation[:, -1]
    owner_cost = last_annual('property_tax') + real_home * (home.maintenance_rate + home.insurance_rate)
    housing_base = np.where(owner, owner_cost, last_annual('housing_cost'))
    ongoing = np.maximum(policy.annual_spending_floor, living + housing_base + policy.annual_healthcare)
    shape = (paths, count)
    housing = np.broadcast_to(housing_base[:, None], shape).copy()
    mortgage = np.zeros(shape)
    pmi = np.zeros(shape)
    mortgage[:, :observed_years] = annual('mortgage_payment')
    housing[:, :observed_years] = annual('housing_cost') - annual('mortgage_interest')

    # First payment is the month after purchase. Use the actual outstanding
    # balance and original contractual term, with a capped final payment.
    balance = series('mortgage')[:, -1]
    purchase_month = np.argmax(series('home_value') > 0, axis=1)
    elapsed = months - 1 - purchase_month
    remaining = np.where(balance > .01, np.maximum(0, home.mortgage_years * 12 - elapsed), 0)
    rate = home.mortgage_rate / 12
    if rate:
        monthly_payment = np.divide(balance * rate, 1 - (1 + rate) ** -remaining,
                                    out=np.zeros(paths), where=remaining > 0)
    else:
        monthly_payment = np.divide(balance, remaining, out=np.zeros(paths), where=remaining > 0)
    future_months = min(int(remaining.max(initial=0)), (count - observed_years) * 12)
    if future_months:
        offset = np.arange(1, future_months + 1)
        growth = (1 + rate) ** (offset - 1)
        opening = (balance[:, None] * growth - monthly_payment[:, None] * (growth - 1) / rate
                   if rate else balance[:, None] - monthly_payment[:, None] * (offset - 1))
        payment = np.minimum(monthly_payment[:, None], np.maximum(0, opening) * (1 + rate))
        payment = np.where(offset <= remaining[:, None], payment, 0.)
        deflator = inflation[:, -1, None] * (1 + policy.future_inflation) ** (offset / 12)
        real_payment = payment / deflator
        closing = np.maximum(0, opening * (1 + rate) - payment)
        projected_home = series('home_value')[:, -1, None] * (1 + policy.future_inflation) ** (offset / 12)
        real_pmi = np.where((closing > .8 * projected_home) & (offset <= remaining[:, None]),
                            closing * home.pmi_rate / 12 / deflator, 0.)
        padded = int(ceil(future_months / 12)) * 12
        for target, source in ((mortgage, real_payment), (pmi, real_pmi)):
            block = np.pad(source, ((0, 0), (0, padded - future_months))).reshape(paths, -1, 12).sum(axis=2)
            target[:, observed_years:observed_years + block.shape[1]] = block

    if scenario is not None:
        births = [(int(year), config.grid.event_month) for year in scenario.births]
    else:
        changes = np.diff(series('children')[0], prepend=0)
        births = [(2026 + (9 + i) // 12, (9 + i) % 12 + 1)
                  for i in np.flatnonzero(changes > 0) for _ in range(int(changes[i]))]
    calendar_months = years[:, None] * 12 + np.arange(12)[None, :]
    minor_counts = np.zeros((count, 12))
    college_counts = np.zeros((count, 12))
    care_units = np.zeros((count, 12))
    for birth_year, birth_month in births:
        age = calendar_months - (birth_year * 12 + birth_month - 1)
        minor_counts += (age >= 0) & (age < policy.college_start_age * 12)
        college_counts += (age >= policy.college_start_age * 12) & (age < policy.child_support_end_age * 12)
        care_units += np.where((age >= 0) & (age < 60), 1., np.where((age >= 60) & (age < 156), config.school_age_childcare_fraction, 0.))
    child = np.broadcast_to(minor_counts.sum(axis=1) * config.household.child_monthly_other, shape).copy()
    college = np.broadcast_to(college_counts.mean(axis=1) * policy.college_annual_cost * policy.college_parent_share, shape).copy()
    final_units = care_units[observed_years - 1, -1]
    care_price = last_annual('childcare') / (12 * final_units) if final_units > 0 else np.zeros(paths)
    childcare = care_price[:, None] * care_units.sum(axis=1)[None, :]
    child[:, :observed_years] = annual('child_other')
    childcare[:, :observed_years] = annual('childcare')
    child_medical = np.broadcast_to(minor_counts.mean(axis=1) * config.household.medical_annual_oop * .5, shape)
    # Adult working benefits persist provisionally. Child OOP support ends at
    # college; the college allowance replaces the student's living-cost budget.
    working_health = last_annual('health_premium') + config.household.medical_annual_oop
    spending = living[:, None] + housing + mortgage + pmi + child + college + childcare + child_medical + working_health[:, None]
    actual = sum(annual(name) for name in ('living_cost', 'housing_cost', 'child_other', 'childcare', 'health_premium', 'medical_out_of_pocket'))
    actual += annual('mortgage_payment') - annual('mortgage_interest')
    spending[:, :observed_years] = actual
    temporary = child + college + childcare + mortgage + pmi + child_medical
    temporary[:, :observed_years] += np.maximum(0, housing[:, :observed_years] - housing_base[:, None])
    # Keep future known birth medical costs and down payments in early-retirement
    # reserves. Cash and home equity stay separate, so down payments are not free.
    medical = annual('medical_out_of_pocket')
    temporary[:, :observed_years] += np.maximum(0, medical - config.household.medical_annual_oop - child_medical[:, :observed_years])
    down = np.where(series('purchased') > 0,
                    (series('home_value') * (1 + home.closing_cost) - series('mortgage')) / inflation, 0.)
    nonrecurring = np.zeros(shape)
    nonrecurring[:, :observed_years] = down[:, 3:].reshape(paths, observed_years, 12).sum(axis=2)
    if scenario is not None:
        if scenario.marriage_year in years:
            nonrecurring[:, scenario.marriage_year - config.start_year] += config.household.wedding_cost
        if scenario.location != 'parents' and scenario.move_year in years:
            nonrecurring[:, scenario.move_year - config.start_year] += home.moving_cost
    temporary += nonrecurring
    if policy.annual_spending_override is not None:
        ongoing[:] = policy.annual_spending_override
        temporary[:] = 0.
    return LifetimeBudget(years, ongoing, temporary, spending, child, college, childcare, mortgage, housing, nonrecurring)
