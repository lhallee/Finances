"""Conditional decision frontiers from paired paths, never pooled scenario trials."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from .compensation import BIO_AI, PHARMA
from .configuration import RunConfig, load_config


def salary_family(career: str) -> str:
    return "bio_ai" if career in BIO_AI else ("pharma_agritech" if career in PHARMA else "other")


def path_diagnostics(result, config: RunConfig, scenario) -> pd.DataFrame:
    """Keep timing and liquidity events distinct from leveraged net worth."""
    from .compensation import starting_salaries
    from .economics import monthly_dates

    dates = monthly_dates(config)[3:]
    cash = result.series("cash")[:, 3:]
    liquid = result.series("liquid_wealth")[:, 3:]
    net = result.series("net_worth")[:, 3:]
    shortfall = result.series("shortfall")[:, 3:] > .01
    depleted = liquid <= .01
    was_positive = np.maximum.accumulate(net > .01, axis=1)
    prior_positive = np.column_stack((np.zeros(len(net), dtype=bool), was_positive[:, :-1]))
    reversal = prior_positive & (net <= 0)
    first = np.argmax(depleted, axis=1)
    has_depleted = depleted.any(axis=1)
    onset = pd.DatetimeIndex(dates[first]).where(has_depleted)
    entry = pd.Timestamp(scenario.career_year, scenario.career_month or config.grid.event_month, 1)
    def within_year(year: int) -> np.ndarray:
        if not year:
            return np.zeros(len(net), dtype=bool)
        elapsed = (onset.year - year) * 12 + onset.month - config.grid.event_month
        return np.asarray(has_depleted & (elapsed >= 0) & (elapsed < 12))
    recent_birth = np.zeros(len(net), dtype=bool)
    for birth in scenario.births:
        recent_birth |= within_year(birth)
    stop_year = scenario.births[scenario.stop_after_birth - 1] if 0 < scenario.stop_after_birth <= len(scenario.births) else 0
    return pd.DataFrame({
        "starting_salary_2026": starting_salaries(config, scenario.career),
        "ever_cash_zero": (cash <= .01).any(axis=1),
        "ever_liquid_depleted": has_depleted,
        "positive_then_negative_net_worth": reversal.any(axis=1),
        "first_liquid_depletion": onset,
        "depletion_before_new_role": has_depleted & (onset < entry),
        "depletion_within_year_of_move": within_year(scenario.move_year if scenario.location != "parents" else 0),
        "depletion_within_year_of_birth": recent_birth,
        "depletion_within_year_of_amanda_stopping": within_year(stop_year),
        "ever_distress": has_depleted | shortfall.any(axis=1),
        "minimum_liquid_wealth_2026": (liquid / result.series("inflation_index")[:, 3:]).min(axis=1),
    })


def frontier_cells(paired: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    """An observation is one independent path, with failure ORed over choices.

    Simultaneous one-sided bounds cover all displayed family/macro/salary cells.
    The event is deliberately conservative: *any* tested choice fails under the
    paired economic draw. No assumption of independent scenario rows is made.
    """
    if "location" not in paired:
        paired = paired.assign(location="all_tested_locations")
    rows = []
    for (family, macro, location), frame in paired.groupby(["salary_family", "macro", "location"], sort=True):
        limits = config.career_salary_ranges[family]
        overrides = [config.career_salary_overrides[c] for c in (*BIO_AI, *PHARMA)
                     if salary_family(c) == family and c in config.career_salary_overrides]
        low = min([limits.minimum, *overrides])
        high = max([limits.maximum, *overrides])
        edges = np.arange(np.floor(low / 50_000) * 50_000, high + 50_000, 50_000)
        for left, right in zip(edges[:-1], edges[1:]):
            cell = frame[(frame.starting_salary_2026 >= left) & (frame.starting_salary_2026 < right)]
            rows.append({"salary_family": family, "macro": macro, "location": location, "salary_low_2026": left,
                         "salary_high_2026": right, "independent_paths": len(cell),
                         "distress_paths": int(cell.ever_distress.sum()),
                         "pre_role_distress_paths": int(cell.depletion_before_new_role.sum()),
                         "tested_scenarios": int(frame.tested_scenarios.max())})
    cells = pd.DataFrame(rows)
    if cells.empty:
        return cells
    alpha = .05 / len(cells)
    n, k = cells.independent_paths.to_numpy(), cells.distress_paths.to_numpy()
    cells["risk_estimate"] = np.divide(k, n, out=np.full(len(n), np.nan), where=n > 0)
    cells["risk_upper_simultaneous_95"] = np.where(k < n, beta.ppf(1-alpha, k+1, n-k), 1.)
    cells["meets_risk_budget"] = (n >= config.decision_minimum_paths) & (cells.risk_upper_simultaneous_95 <= config.decision_risk_budget)
    cells["supported_threshold_2026"] = np.nan
    for _, group in cells.groupby(["salary_family", "macro", "location"]):
        ordered = group.sort_values("salary_low_2026")
        all_higher_pass = ordered.meets_risk_budget.iloc[::-1].cummin().iloc[::-1]
        passed = ordered.loc[all_higher_pass]
        if len(passed):
            cells.loc[group.index, "supported_threshold_2026"] = passed.salary_low_2026.min()
    return cells


def generate_decision_report(folder: Path, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream terminal partitions, retaining only paired event flags in memory."""
    manifest = json.loads((folder / "manifest.json").read_text())
    config = load_config(folder / "config.json")
    columns = ["scenario_id", "path_id", "starting_salary_2026", "ever_distress", "depletion_before_new_role"]
    catalog = summary[["scenario_id", "career", "macro"]].copy()
    catalog["location"] = summary["location"] if "location" in summary else "unspecified"
    catalog["salary_family"] = catalog.career.map(salary_family)
    catalog = catalog[catalog.salary_family != "other"]
    catalog = pd.concat([catalog, catalog.assign(macro="all_tested_macros"),
                         catalog.assign(location="all_tested_locations"),
                         catalog.assign(macro="all_tested_macros", location="all_tested_locations")], ignore_index=True)
    catalog = catalog.drop_duplicates(["scenario_id", "macro", "location"])
    retained = None
    keys = ["salary_family", "macro", "location", "path_id", "starting_salary_2026"]
    for part in manifest["parts"]:
        frame = pd.read_parquet(folder / "terminal" / part["file"], columns=columns)
        frame = frame.merge(catalog, on="scenario_id", validate="many_to_many")
        if frame.empty:
            continue
        reduced = frame.groupby(keys, as_index=False).agg(
            ever_distress=("ever_distress", "max"), depletion_before_new_role=("depletion_before_new_role", "max"))
        retained = reduced if retained is None else pd.concat([retained, reduced]).groupby(keys, as_index=False).max()
    if retained is None:
        cells = pd.DataFrame()
    else:
        counts = catalog.groupby(["salary_family", "macro", "location"]).scenario_id.nunique()
        retained["tested_scenarios"] = [counts[f, m, loc] for f, m, loc in
                                         retained[["salary_family", "macro", "location"]].itertuples(index=False, name=None)]
        cells = frontier_cells(retained, config)
    cells.to_csv(folder / "salary_risk_frontier.csv", index=False)
    fields = [c for c in ("scenario_id", "career", "location", "housing", "career_year", "move_year", "marriage_year",
                         "births", "stop_after_birth", "macro", "grant_case", "liquid_depletion_probability",
                         "cash_zero_probability", "net_worth_reversal_probability", "distress_probability",
                         "distress_mc_low", "distress_mc_high", "depletion_before_role_probability",
                         "depleted_paths", "recent_move_share_of_depletions", "recent_birth_share_of_depletions",
                         "recent_amanda_stop_share_of_depletions") if c in summary]
    risks = summary[fields].sort_values("distress_probability", ascending=False)
    risks.to_csv(folder / "depletion_combinations.csv", index=False)
    text = ["# Conditional life-choice analysis", "",
            f"Coverage: {manifest['completed_scenarios']:,}/{manifest['expected_scenarios']:,} configured scenarios. Risk budget: {config.decision_risk_budget:.0%}.", "",
            "Distress means exhausted liquid wealth after tax liabilities/unpaid obligations, or an unmet bill. Cash at zero and a positive-to-negative net-worth reversal are reported separately.", "",
            "Salary is annual starting pay in 2026 dollars. Each salary band follows the configured salary distribution inside that band, not every exact dollar value. The frontier ORs distress across every tested choice for each paired path, including failure before the new role starts. It never counts correlated scenario rows as independent trials.", "",
            "Upper bounds are one-sided exact binomial bounds with a Bonferroni correction across all reported salary cells. A threshold requires enough independent paths and every tested higher band to meet the risk budget. Empty or noisy upper-tail bands prevent certification. No interpolation or extrapolation beyond tested salaries is used.", "",
            "These are conditional liquidity screens, not a utility judgment or a guarantee. They do not establish that every location or birth schedule is covered unless that combination was configured and completed. A missing threshold can mean adverse outcomes or insufficient precision; it is not proof that no salary works.", ""]
    if not cells.empty:
        for (family, macro, location), group in cells.groupby(["salary_family", "macro", "location"]):
            if location != "all_tested_locations":
                continue
            threshold = group.supported_threshold_2026.iloc[0]
            message = f"Supported tested-band threshold: ${threshold:,.0f}" if pd.notna(threshold) else "No supported salary threshold in this run"
            pre_role = group.pre_role_distress_paths.sum() / max(1, group.independent_paths.sum())
            text.append(f"- {family.replace('_', ' ').title()}, {macro.replace('_', ' ')}: {message}. {pre_role:.1%} of paired paths had liquid depletion before the new role in at least one tested choice configuration.")
    text.extend(["", "## Highest-risk tested combinations", "",
                 "Event shares describe the first liquid-depletion month occurring within 12 months of a scheduled event. Events can overlap; temporal proximity does not establish cause.", ""])
    for row in risks.head(10).itertuples():
        if not hasattr(row, "location"):
            break
        text.append(f"- {row.career.replace('_', ' ').title()}; {row.location.replace('_', ' ')}; {row.housing}; "
                    f"births {row.births or 'none'}; Amanda stop after birth {row.stop_after_birth}; {row.macro}; "
                    f"new role {row.career_year}, move {row.move_year}: distress {row.distress_probability:.1%} "
                    f"(95% Monte Carlo interval {row.distress_mc_low:.1%} to {row.distress_mc_high:.1%}).")
    (folder / "decision_analysis.md").write_text("\n".join(text), encoding="utf-8")
    return risks, cells
