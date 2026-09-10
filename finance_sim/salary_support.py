"""Salary-only counterfactuals with paired liquidity, reserve and retirement tests."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import shutil
import time
import zipfile
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from scipy.stats import beta

from .business import BUSINESS
from .configuration import RunConfig, config_dict
from .economics import economic_paths, monthly_dates, history_fingerprint
from .engine import METRICS, simulate
from .results import SimulationResult
from .retirement import project_retirement
from .scenarios import Scenario, practical_families
from .storage import code_fingerprint, write_json


def support_scenarios(config: RunConfig) -> list[Scenario]:
    """Every declared location, macro, move date and applicable stay-home policy."""
    policy = config.salary_support
    if policy.company_support not in ("none", "darpa_only"):
        raise ValueError("Company support must be none or darpa_only")
    transition = pd.Timestamp(policy.salary_start_year, config.grid.event_month, 1)
    darpa_only = policy.company_support == "darpa_only"
    if darpa_only:
        grant = next((g for g in config.business.grants if g.name == "darpa"), None)
        if grant is None:
            raise ValueError("DARPA-only support requires a DARPA grant")
        transition = pd.Timestamp(grant.start) + pd.DateOffset(months=grant.months)
        if transition.year > config.end_year:
            raise ValueError("DARPA must end within the simulated horizon")
    scenarios = []
    for location in dict.fromkeys(config.grid.locations):
        homes = ("free",) if location == "parents" else tuple(dict.fromkeys(config.grid.housing))
        moves = (policy.salary_start_year,) if location == "parents" else tuple(dict.fromkeys(config.grid.practical_transition_years))
        for home, move, macro, (births, stop) in itertools.product(homes, moves, config.grid.macros, practical_families(config)):
            if not stop:
                continue
            scenarios.append(Scenario(career="bio_ai_engineer", career_year=transition.year,
                                      location=location, housing=home, move_year=move, marriage_year=policy.marriage_year,
                                      macro=macro, births=births, stop_after_birth=stop, company_mode="direct",
                                      company_outcome="dormant", grants=("darpa",) if darpa_only else (), exit_value=0, exit_year=0,
                                      prior_career="ud_blend" if darpa_only else "ud", career_month=transition.month))
    return list(dict.fromkeys(scenarios))


def group_keys(scenario: Scenario) -> list[tuple[str, str]]:
    return [("overall", "all"), ("location", scenario.location), ("children", str(len(scenario.births))),
            ("macro", scenario.macro), ("stop_after_birth", str(scenario.stop_after_birth)),
            ("housing", scenario.housing), ("move_year", "not_applicable" if scenario.location == "parents" else str(scenario.move_year))]


def support_outcomes(result: SimulationResult, config: RunConfig, scenario: Scenario) -> dict[str, np.ndarray]:
    """Monthly constraints use actual ledgers, never subtraction from taxed outputs."""
    channels = (BUSINESS.distribution, BUSINESS.equity_proceeds, BUSINESS.ordinary_proceeds)
    if config.salary_support.company_support == "none":
        channels = (BUSINESS.founder_pay, *channels)
    if np.any(np.abs(result.business[:, :, channels]) > 1e-9):
        raise ArithmeticError("The salary-only counterfactual contains personal Synthyra receipts")
    if config.salary_support.company_support == "darpa_only":
        grant = next(g for g in config.business.grants if g.name == "darpa")
        dates_all = monthly_dates(config)
        start = pd.Timestamp(grant.start)
        end_grant = start + pd.DateOffset(months=grant.months)
        outside = (dates_all < start) | (dates_all >= end_grant)
        if np.any(np.abs(result.business[:, outside, BUSINESS.founder_pay]) > 1e-9):
            raise ArithmeticError("DARPA-only compensation extends outside the award period")
    liquid = result.series("liquid_wealth")[:, 3:]
    inflation = result.series("inflation_index")[:, 3:]
    real_liquid = liquid / inflation
    distress = (liquid <= .01) | (result.series("shortfall")[:, 3:] > .01)
    # Essential cash spending includes contractual loan payments, not voluntary
    # extra debt payments or savings. Taxes payable already reduce liquid wealth.
    costs = sum(result.series(name) for name in ("living_cost", "childcare", "child_other", "health_premium", "housing_cost", "medical_out_of_pocket"))
    costs += np.maximum(0, result.series("debt_payment") - result.series("debt_extra"))
    costs += np.maximum(0, result.series("mortgage_payment") - result.series("mortgage_interest"))
    real_costs = costs / result.series("inflation_index")
    cumulative = np.column_stack((np.zeros(len(costs)), np.cumsum(real_costs, axis=1)))
    end = np.arange(4, real_costs.shape[1] + 1)
    start = np.maximum(0, end - 12)
    spending = (cumulative[:, end] - cumulative[:, start]) / (end - start)
    active = result.series("amanda_stopped")[:, 3:] > .5
    reserve_gap = real_liquid - spending * config.salary_support.reserve_months
    retirement = project_retirement(result.household, config, scenario).minimum_retirement_age.fillna(np.inf).to_numpy()
    dates = monthly_dates(config)[3:]
    before = dates < pd.Timestamp(scenario.career_year, scenario.career_month or config.grid.event_month, 1)
    return {"distress": distress.any(axis=1), "reserve_failure": ((reserve_gap < -.01) & active).any(axis=1),
            "pre_salary_distress": distress[:, before].any(axis=1), "retirement_age": retirement,
            "minimum_liquid_2026": real_liquid.min(axis=1),
            "minimum_reserve_margin_2026": np.where(active, reserve_gap, np.inf).min(axis=1)}


def support_statistics(paths: pd.DataFrame, retirement_age: float, risk_budget: float = .05,
                       requirements: tuple[str, ...] = ("retirement", "bills", "reserve"),
                       minimum_paths: int = 32) -> pd.DataFrame:
    """One independent trial per paired path, ORed over every choice in its scope."""
    frame = paths.assign(retirement_failure=paths.retirement_age > retirement_age)
    if not requirements or set(requirements) - {"retirement", "bills", "reserve"}:
        raise ValueError("Select one or more supported requirements")
    columns = {"retirement": "retirement_failure", "bills": "distress", "reserve": "reserve_failure"}
    frame["failure"] = frame[[columns[key] for key in requirements]].any(axis=1)
    keys = ["dimension", "choice", "salary_2026"]
    grouped = frame.groupby(keys, sort=True)
    stats = grouped.agg(independent_paths=("path_id", "size"), failures=("failure", "sum"),
                        depletion_probability=("distress", "mean"), reserve_failure_probability=("reserve_failure", "mean"),
                        retirement_failure_probability=("retirement_failure", "mean"),
                        pre_salary_failure_probability=("pre_salary_distress", "mean"),
                        median_minimum_liquid_2026=("minimum_liquid_2026", "median"),
                        median_reserve_margin_2026=("minimum_reserve_margin_2026", "median"),
                        tested_configurations=("tested_configurations", "max")).reset_index()
    if stats.empty:
        return stats
    n, k = stats.independent_paths.to_numpy(), stats.failures.to_numpy()
    alpha = .05 / len(stats)
    upper = np.where(k < n, beta.ppf(1-alpha, k+1, n-k), 1.)
    stats["success_probability"] = 1-k/n
    stats["success_lower_simultaneous_95"] = 1-upper
    stats["meets_target"] = (upper <= risk_budget) & (n >= minimum_paths)
    stats["supported_salary_2026"] = np.nan
    for _, group in stats.groupby(["dimension", "choice"]):
        ordered = group.sort_values("salary_2026")
        passes = ordered.meets_target.iloc[::-1].cummin().iloc[::-1]
        qualifying = ordered.loc[passes]
        if len(qualifying):
            stats.loc[group.index, "supported_salary_2026"] = qualifying.salary_2026.min()
    return stats


def run_salary_support(config: RunConfig, folder: Path, resume: bool = False, progress=print) -> dict:
    config = dataclasses.replace(config, business=dataclasses.replace(config.business, monthly_revenue=0., distributions_fraction=0.,
                                 grants=[dataclasses.replace(g, follow_on_probability=0., follow_on_annual_revenue=0.) for g in config.business.grants]),
                                 career_salary_overrides=dict(config.career_salary_overrides))
    config.validate()
    policy = config.salary_support
    if policy.retirement_age > config.retirement.maximum_retirement_age:
        raise ValueError("Retirement projection must extend at least to the support target age")
    if policy.salary_start_year < config.start_year or policy.salary_start_year > config.end_year:
        raise ValueError("Salary start must fall within the simulated horizon")
    cases = support_scenarios(config)
    if not cases:
        raise ValueError("No applicable birth and stay-home choices were configured")
    groups = sorted({key for s in cases for key in group_keys(s)})
    group_index = {key: index for index, key in enumerate(groups)}
    sizes = {key: sum(key in group_keys(s) for s in cases) for key in groups}
    fingerprint = hashlib.sha256(json.dumps({"config": config_dict(config), "history": history_fingerprint(config),
                                            "engine": code_fingerprint()}, sort_keys=True).encode()).hexdigest()
    manifest_path = folder / "salary_support.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not resume:
            raise FileExistsError("Choose a new output folder or use --resume")
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Cannot resume with different inputs or engine")
        for part in manifest["parts"]:
            for key in ("paths", "cases"):
                if hashlib.sha256((folder / part[key]).read_bytes()).hexdigest() != part[f"{key}_hash"]:
                    raise ValueError("A salary-support checkpoint is corrupted")
        if manifest["status"] == "complete":
            return manifest
    else:
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError("Choose an empty output folder")
        folder.mkdir(parents=True, exist_ok=True)
        resolved = config_dict(config)
        if config.economics.history_csv:
            shutil.copy2(config.economics.history_csv, folder / "history.csv")
            resolved["economics"]["history_csv"] = "history.csv"
        write_json(folder / "config.json", resolved)
        from .reference import SOURCES
        write_json(folder / "sources.json", SOURCES)
        with zipfile.ZipFile(folder / "engine_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for source in sorted(Path(__file__).parent.glob("*.py")):
                archive.write(source, f"finance_sim/{source.name}")
        manifest = {"kind": "salary_support", "status": "running", "fingerprint": fingerprint,
                    "retirement_budget_model": "lifecycle_v3",
                    "company_support": policy.company_support,
                    "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "retirement_age": policy.retirement_age,
                    "retirement_funding_target": config.retirement.success_target, "retirement_life_expectancy": config.retirement.life_expectancy,
                    "risk_budget": config.decision_risk_budget, "minimum_paths": config.decision_minimum_paths, "reserve_months": policy.reserve_months,
                    "paths_per_configuration": config.paths, "salaries_2026": list(policy.salaries),
                    "expected_configurations": len(cases), "expected_salary_configurations": len(cases)*len(policy.salaries),
                    "completed_salary_configurations": 0, "parts": [], "runtime_seconds": 0.,
                    "coverage": {"locations": sorted({s.location for s in cases}), "macros": sorted({s.macro for s in cases}),
                                 "birth_schedules": sorted({s.births for s in cases}), "move_years": sorted({s.move_year for s in cases if s.location != "parents"}),
                                 "marriage_year": policy.marriage_year, "salary_start_year": cases[0].career_year,
                                 "salary_start_month": cases[0].career_month or config.grid.event_month,
                                 "child_support_end_age": config.retirement.child_support_end_age,
                                 "college_annual_cost": config.retirement.college_annual_cost,
                                 "college_parent_share": config.retirement.college_parent_share,
                                 "mortgage_years": config.housing.mortgage_years,
                                 "household_benefits": "Standard modeled employee benefits; family/replacement coverage according to eligibility"},
                    "notice": "Conditional tests of configured choices, not every possible birth date or future economy. Salary points are hypotheses, not offer probabilities. Reserves are tested after Amanda stops; unpaid bills/depletion are tested over the whole horizon. Retirement uses the existing beyond-2036 projection, not a new lifetime career simulation."}
        write_json(manifest_path, manifest)
    numba.set_num_threads(min(config.threads, numba.config.NUMBA_NUM_THREADS))
    draws = {macro: economic_paths(config, macro) for macro in {s.macro for s in cases}}
    start = time.perf_counter()
    prior_time = manifest["runtime_seconds"]
    for salary in policy.salaries[len(manifest["parts"]):]:
        config.career_salary_overrides["bio_ai_engineer"] = salary
        shape = (len(groups), config.paths)
        aggregates = {name: np.zeros(shape, dtype=bool) for name in ("distress", "reserve_failure", "pre_salary_distress")}
        aggregates["retirement_age"] = np.zeros(shape)
        for name in ("minimum_liquid_2026", "minimum_reserve_margin_2026"):
            aggregates[name] = np.full(shape, np.inf)
        case_rows = []
        for index, scenario in enumerate(cases):
            result = SimulationResult(*simulate(config, scenario, draws[scenario.macro]))
            outcome = support_outcomes(result, config, scenario)
            for group in group_keys(scenario):
                row = group_index[group]
                for name, values in outcome.items():
                    if name.startswith("minimum_"):
                        aggregates[name][row] = np.minimum(aggregates[name][row], values)
                    else:
                        aggregates[name][row] = np.maximum(aggregates[name][row], values)
            success = ~(outcome["distress"] | outcome["reserve_failure"] | (outcome["retirement_age"] > policy.retirement_age))
            case_rows.append({**scenario.record(), "salary_2026": salary, "success_probability": success.mean(),
                              "distress_probability": outcome["distress"].mean(), "reserve_failure_probability": outcome["reserve_failure"].mean(),
                              "retirement_by_target_probability": (outcome["retirement_age"] <= policy.retirement_age).mean()})
            if (index+1) % 500 == 0:
                progress(f"${salary:,.0f}: {index+1:,}/{len(cases):,} life configurations", flush=True)
        frames = [pd.DataFrame({"dimension": dimension, "choice": choice, "salary_2026": salary,
                                "path_id": np.arange(config.paths), "tested_configurations": sizes[dimension, choice],
                                **{name: values[i] for name, values in aggregates.items()}})
                  for i, (dimension, choice) in enumerate(groups)]
        part = {}
        number = len(manifest["parts"])
        for kind, frame in (("paths", pd.concat(frames, ignore_index=True)), ("cases", pd.DataFrame(case_rows))):
            filename = f"{kind}-{number:03d}.parquet"
            temporary = folder / f"{filename}.tmp"
            frame.to_parquet(temporary, index=False)
            temporary.replace(folder / filename)
            part[kind] = filename
            part[f"{kind}_hash"] = hashlib.sha256((folder / filename).read_bytes()).hexdigest()
        manifest["parts"].append(part)
        manifest["completed_salary_configurations"] += len(cases)
        manifest["runtime_seconds"] = prior_time + time.perf_counter()-start
        write_json(manifest_path, manifest)
        progress(f"Saved ${salary:,.0f}: {manifest['completed_salary_configurations']:,}/{manifest['expected_salary_configurations']:,} salary/configuration tests", flush=True)
    retained = pd.concat([pd.read_parquet(folder / part["paths"]) for part in manifest["parts"]], ignore_index=True)
    stats = support_statistics(retained, policy.retirement_age, config.decision_risk_budget,
                               minimum_paths=config.decision_minimum_paths)
    stats.to_csv(folder / "support_trends.csv", index=False)
    manifest["status"] = "complete"
    manifest["runtime_seconds"] = prior_time + time.perf_counter()-start
    manifest["completed_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    write_json(manifest_path, manifest)
    return manifest
