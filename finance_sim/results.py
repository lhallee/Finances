"""Named result access and distribution summaries. Percentiles are not ledger paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass
from scipy.stats import beta

from .business import BUSINESS_COLUMNS
from .configuration import RunConfig
from .economics import monthly_dates
from .engine import METRICS
from .scenarios import Scenario
from .retirement import project_retirement
from .decisions import path_diagnostics
from .scenarios import compatible


@dataclass(frozen=True)
class SimulationResult:
    household: np.ndarray  # (paths, months, household metrics)
    business: np.ndarray  # (paths, months, business metrics)

    def series(self, name: str) -> np.ndarray:
        if name in METRICS:
            return self.household[:, :, METRICS.index(name)]  # (paths, months)
        return self.business[:, :, BUSINESS_COLUMNS.index(name)]  # (paths, months)

    def ledger(self, path: int, config: RunConfig) -> pd.DataFrame:
        ledger = pd.DataFrame(self.household[path], columns=METRICS, index=monthly_dates(config))
        for column, name in enumerate(BUSINESS_COLUMNS):
            ledger[f"business_{name}"] = self.business[path, :, column]  # (months,)
        ledger.index.name = "date"
        return ledger.reset_index()


def probability_interval(count: int, n: int) -> tuple[float, float]:
    """Exact binomial 95% interval for Monte Carlo event-probability error."""
    return (float(beta.ppf(.025, count, n-count+1)) if count else 0.,
            float(beta.ppf(.975, count+1, n-count)) if count < n else 1.)


def summarize(result: SimulationResult, config: RunConfig, scenario: Scenario) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    values = result.household[:, 3:, :]  # (p,120,k), excludes bridge.
    p = len(values)
    terminal = pd.DataFrame(values[:, -1], columns=METRICS)
    terminal["path_id"] = np.arange(p)  # (p,)
    terminal["scenario_id"] = scenario.id
    terminal["ever_shortfall"] = np.any(result.series("shortfall")[:, 3:] > .01, axis=1)  # (p,)
    terminal["ever_bought"] = np.any(result.series("purchased")[:, 3:] > 0, axis=1)  # (p,)
    terminal["cumulative_tax"] = result.series("tax_expense")[:, 3:].sum(axis=1)  # (p,)
    for name in ("amanda_income", "childcare", "health_premium", "amanda_employer_retirement", "earned_income", "exit_cash", "debt_interest"):
        terminal[f"cumulative_{name}"] = result.series(name)[:, 3:].sum(axis=1)  # (p,)
    monetary = ("net_worth", "financial_net_worth", "liquid_wealth", "cash", "brokerage", "retirement", "debt", "home_equity", "tax_expense", "childcare", "health_premium", "earned_income", "business_interest", "unpaid_bills")
    for name in monetary:
        terminal[f"real_{name}"] = terminal[name] / terminal.inflation_index
    for name in ("amanda_income", "childcare", "health_premium", "amanda_employer_retirement", "earned_income", "exit_cash", "debt_interest", "tax"):
        metric = "tax_expense" if name == "tax" else name
        terminal[f"real_cumulative_{name}"] = (result.series(metric)[:, 3:] / result.series("inflation_index")[:, 3:]).sum(axis=1)
    terminal = pd.concat([terminal, path_diagnostics(result, config, scenario)], axis=1)
    retirement = project_retirement(result.household, config, scenario)
    terminal = pd.concat([terminal, retirement], axis=1)
    quantiles = (.025, .10, .25, .5, .75, .90, .975)
    selected = ("net_worth", "financial_net_worth", "liquid_wealth", "cash", "brokerage", "retirement", "debt", "home_equity", "tax_expense", "childcare", "health_premium", "earned_income", "business_interest", "unpaid_bills")
    frames = []
    dates = monthly_dates(config)[3:]
    for metric in selected:
        paths = result.series(metric)[:, 3:] / result.series("inflation_index")[:, 3:]  # (p,120)
        bands = np.quantile(paths, quantiles, axis=0)  # (7,120)
        frame = pd.DataFrame({"scenario_id": scenario.id, "date": dates, "metric": metric,
                              **{f"q{int(q*1000):03d}": bands[i] for i,q in enumerate(quantiles)},
                              "mean": paths.mean(axis=0)})
        frames.append(frame)
    summary = scenario.record()
    summary["births"] = ",".join(map(str, scenario.births))
    summary["grants"] = ",".join(scenario.grants)
    summary["children"] = len(scenario.births)
    summary["paths"] = p
    summary["dollar_basis"] = "2026 USD"
    summary["employer_location_permission_unresolved"] = not compatible(scenario.career, scenario.location)
    for label, column in (("liquid_depletion", "ever_liquid_depleted"), ("cash_zero", "ever_cash_zero"),
                          ("net_worth_reversal", "positive_then_negative_net_worth"), ("distress", "ever_distress"),
                          ("depletion_before_role", "depletion_before_new_role")):
        summary[f"{label}_probability"] = float(terminal[column].mean())
    summary["distress_mc_low"], summary["distress_mc_high"] = probability_interval(int(terminal.ever_distress.sum()), p)
    depleted = terminal[terminal.ever_liquid_depleted]
    summary["depleted_paths"] = len(depleted)
    for label, column in (("move", "depletion_within_year_of_move"), ("birth", "depletion_within_year_of_birth"),
                          ("amanda_stop", "depletion_within_year_of_amanda_stopping")):
        summary[f"recent_{label}_share_of_depletions"] = float(depleted[column].mean()) if len(depleted) else np.nan
    summary["median_net_worth"] = float(terminal.real_net_worth.median())
    summary["mean_net_worth"] = float(terminal.real_net_worth.mean())
    summary["mean_mc_se"] = float(terminal.real_net_worth.std(ddof=1) / np.sqrt(p))
    summary["p025_net_worth"] = float(terminal.real_net_worth.quantile(.025))
    summary["p975_net_worth"] = float(terminal.real_net_worth.quantile(.975))
    summary["shortfall_probability"] = float(terminal.ever_shortfall.mean())
    low, high = probability_interval(int(terminal.ever_shortfall.sum()), p)
    summary["shortfall_mc_low"], summary["shortfall_mc_high"] = low, high
    summary["buy_probability"] = float(terminal.ever_bought.mean())
    summary["tail_warning"] = p * .025 < 20
    reached = retirement.minimum_retirement_age.notna()
    # Include censored paths as infinity before taking quantiles. Do not report
    # an artificially early median by dropping paths that never qualify.
    retirement_ages = retirement.minimum_retirement_age.fillna(np.inf).to_numpy()
    for label, quantile in (("p10", .1), ("median", .5), ("p90", .9)):
        value = float(np.quantile(retirement_ages, quantile, method="inverted_cdf"))
        summary[f"{label}_minimum_retirement_age"] = value if np.isfinite(value) else np.nan
    summary["retirement_reached_probability"] = float(reached.mean())
    summary["retirement_within_horizon_probability"] = float((retirement.retirement_age_source == "within_simulation").mean())
    summary["retirement_success_target"] = config.retirement.success_target
    summary["retirement_maximum_age"] = config.retirement.maximum_retirement_age
    for name in ("amanda_income", "childcare", "health_premium", "amanda_employer_retirement", "earned_income", "exit_cash", "debt_interest", "tax"):
        summary[f"mean_cumulative_{name}"] = float(terminal[f"real_cumulative_{name}"].mean())
    return pd.DataFrame([summary]), terminal, pd.concat(frames, ignore_index=True)


def paired_amanda_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    keys = [name for name in ("career", "career_year", "location", "move_year", "housing", "marriage_year", "births", "amanda_career", "company_mode", "company_outcome", "macro", "benefits", "grants", "grant_case", "exit_value", "exit_year", "exit_type") if name in summary]
    working = summary[summary.stop_after_birth == 0]
    stopped = summary[summary.stop_after_birth > 0]
    merged = stopped.merge(working, on=keys, suffixes=("_stay_home", "_working"), validate="many_to_one")
    for label, column in (("net_worth_difference", "median_net_worth"), ("income_difference", "mean_cumulative_amanda_income"),
                          ("childcare_difference", "mean_cumulative_childcare"), ("insurance_difference", "mean_cumulative_health_premium"),
                          ("employer_retirement_difference", "mean_cumulative_amanda_employer_retirement"), ("tax_difference", "mean_cumulative_tax")):
        merged[label] = merged[f"{column}_stay_home"] - merged[f"{column}_working"]
    return merged
