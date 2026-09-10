"""Retirement readiness diagnostic, separate from the working household ledger."""

from functools import lru_cache
from math import ceil

import numpy as np
import pandas as pd

from .configuration import RunConfig
from .engine import METRICS
from .lifetime_budget import lifetime_budget


@lru_cache(maxsize=32)
def capital_factors(seed: int, paths: int, years: int, mean: float, volatility: float,
                    target: float) -> np.ndarray:
    """Capital per annual dollar spent at the beginning of each retirement year.

    Both the lifetime and accessible-account bridge use (1+target)/2 quantiles.
    Bonferroni bounds their combined modeled failure probability by 1-target.
    Cached common draws keep comparison noise and computation small.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, 719]))
    growth = np.exp(np.log1p(mean) - volatility**2 / 2 +
                    volatility * rng.standard_normal((paths, years)))  # (draws, years)
    discount = np.column_stack((np.ones(paths), 1 / np.cumprod(growth[:, :-1], axis=1)))  # (draws, years)
    annuity = np.cumsum(discount, axis=1)  # (draws, years)
    factors = np.r_[0., np.quantile(annuity, (1 + target) / 2, axis=0, method="higher")]  # (years+1,)
    factors.setflags(write=False)
    return factors


def project_retirement(household: np.ndarray, config: RunConfig, scenario=None) -> pd.DataFrame:
    """Return earliest qualifying Logan age per path, or NaN if not reached.

    Annual candidates use actual year-end balances through the saved horizon.
    Later candidates use stochastic real returns and the median of the last
    three annual recurring saving amounts. There are no future exits or grants.
    Home equity and unsold company interests never fund retirement here.
    """
    policy = config.retirement
    paths, months = household.shape[:2]
    def series(name: str) -> np.ndarray:
        return household[:, :, METRICS.index(name)]  # (paths, months)

    inflation = series("inflation_index")
    years = (months - 3) // 12
    ends = np.arange(14, months, 12)  # (years,)
    ages = config.household.logan.age + (ends + 1) / 12  # (years,)
    # Subtract outstanding loans and a flat reserve for taxable gains. Household
    # liquid_wealth already deducts tax payable and unpaid bills.
    accessible = (series("liquid_wealth") - series("debt") -
                  np.maximum(0, series("brokerage") - series("brokerage_basis")) * policy.withdrawal_tax_rate) / inflation
    deferred = series("retirement") * (1 - policy.withdrawal_tax_rate) / inflation
    budget = lifetime_budget(household, config, scenario)
    # Finite obligations are earmarked at zero real return. They are not treated
    # as perpetual portfolio withdrawals, and no risky return is needed to pay
    # college or the remaining mortgage. The ongoing adult budget is funded by
    # the stochastic portfolio. This is a conservative, explicit two-bucket plan.
    remaining_costs = np.cumsum(budget.temporary_retirement[:, ::-1], axis=1)[:, ::-1]
    remaining_costs = np.column_stack((remaining_costs, np.zeros(paths)))
    # Life expectancy applies to the younger adult; ages are reported for Logan.
    end_age = policy.life_expectancy + max(0, config.household.logan.age - config.household.amanda.age)
    factors = capital_factors(config.seed, policy.retirement_paths,
                              ceil(end_age - config.household.logan.age), policy.real_return,
                              policy.annual_volatility, policy.success_target)
    earliest = np.full(paths, np.nan)  # (paths,)
    origin = np.full(paths, "not_reached", dtype=object)  # (paths,)
    budget_at_retirement = np.full(paths, np.nan)  # (paths,)
    required_at_retirement = np.full(paths, np.nan)  # (paths,)
    access_age = policy.account_access_age + max(0, config.household.logan.age - config.household.amanda.age)

    def evaluate(age: float, liquid: np.ndarray, locked: np.ndarray, next_year: int, source: str) -> None:
        if age > policy.maximum_retirement_age:
            return
        lifetime = ceil(end_age - age)
        bridge = max(0, ceil(access_age - age))
        transient = remaining_costs[:, next_year]
        required = budget.ongoing_retirement * max(1 / policy.withdrawal_rate, factors[lifetime]) + transient
        bridge_end = min(next_year + bridge, len(budget.years))
        bridge_cost = budget.ongoing_retirement * factors[bridge] + transient - remaining_costs[:, bridge_end]
        eligible = (liquid + locked >= required) & ((bridge == 0) | (liquid >= bridge_cost)) & np.isnan(earliest)
        earliest[eligible] = age
        origin[eligible] = source
        annual = budget.ongoing_retirement + budget.temporary_retirement[:, next_year]
        budget_at_retirement[eligible] = annual[eligible]
        required_at_retirement[eligible] = required[eligible]

    for year, month in enumerate(ends):
        evaluate(float(ages[year]), accessible[:, month], deferred[:, month], year + 1, "within_simulation")
    # Cash additions include drawdowns; subtract liquidity-event inflows to avoid
    # projecting a one-time exit as an annual saving rate for the rest of life.
    cash_change = np.diff(series("cash"), axis=1, prepend=series("cash")[:, :1])
    prior_broker = np.column_stack((np.full(paths, config.household.logan.brokerage + config.household.amanda.brokerage), series('brokerage')[:, :-1]))
    prior_retirement = np.column_stack((np.full(paths, config.household.logan.retirement + config.household.amanda.retirement), series('retirement')[:, :-1]))
    taxable_fraction = np.divide(prior_broker, prior_broker + prior_retirement, out=np.zeros_like(prior_broker), where=prior_broker + prior_retirement > 0)
    broker_flow = series('brokerage') - prior_broker - series('investment_return') * taxable_fraction
    saving = cash_change - series("cash_interest") - series("exit_cash") + broker_flow
    saving += (series("employee_retirement") + series("employer_retirement")) * (1 - policy.withdrawal_tax_rate)
    annual_saving = (saving[:, 3:] / inflation[:, 3:]).reshape(paths, years, 12).sum(axis=2)  # (paths, years)
    recurring_saving = np.median(annual_saving[:, -min(3, years):], axis=1)  # (paths,)
    # Outstanding non-mortgage debt was reserved in accessible wealth above.
    # Add its historical payments back to the post-horizon spending capacity.
    debt_paid = (series('debt_payment')[:, 3:] / inflation[:, 3:]).reshape(paths, years, 12).sum(axis=2)
    resources = np.median((annual_saving + budget.working_spending[:, :years] + debt_paid + budget.nonrecurring[:, :years])[:, -min(3, years):], axis=1)
    locked_saving = ((series("employee_retirement") + series("employer_retirement"))[:, 3:] /
                     inflation[:, 3:] * (1 - policy.withdrawal_tax_rate)).reshape(paths, years, 12).sum(axis=2)
    recurring_locked = np.median(locked_saving[:, -min(3, years):], axis=1)  # (paths,)
    liquid, locked = accessible[:, -1].copy(), deferred[:, -1].copy()  # each (paths,)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 811]))
    if policy.project_beyond_horizon:
        for step in range(1, max(0, int(policy.maximum_retirement_age - ages[-1])) + 1):
            growth = np.exp(np.log1p(policy.real_return) - policy.annual_volatility**2 / 2 +
                            policy.annual_volatility * rng.standard_normal(paths))  # (paths,)
            year_index = years + step - 1
            adjusted_saving = resources - budget.working_spending[:, year_index]
            liquid = liquid * growth + adjusted_saving - recurring_locked
            locked = locked * growth + recurring_locked
            evaluate(float(ages[-1] + step), liquid, locked, year_index + 1, "beyond_horizon_projection")
    return pd.DataFrame({"minimum_retirement_age": earliest, "retirement_age_source": origin,
                         "retirement_annual_spending_real": budget_at_retirement,
                         "retirement_required_capital_real": required_at_retirement,
                         "projected_annual_saving_real": recurring_saving,
                         "projected_annual_saving_after_obligations_real": resources - budget.working_spending[:, -1],
                         "retirement_budget_model": "lifecycle_v2"})
