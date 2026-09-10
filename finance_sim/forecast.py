"""Subjective weighted forecast, kept separate from conditional scenario mixtures."""

import dataclasses
import json

import numpy as np
import pandas as pd

from pathlib import Path

from .configuration import RunConfig
from .economics import economic_paths
from .engine import simulate
from .reference import MACROS
from .results import SimulationResult, summarize
from .scenarios import Scenario
from .storage import write_json


def run_forecast(config: RunConfig, folder: Path) -> pd.DataFrame:
    if config.forecast_paths < 2:
        raise ValueError("forecast_paths must be at least 2")
    careers = list(config.forecast_career_weights)
    career_weights = np.asarray(list(config.forecast_career_weights.values()), dtype=float)  # (c,)
    macro_weights = np.asarray(config.macro_weights, dtype=float)  # (6,)
    if np.any(career_weights < 0) or np.any(macro_weights < 0) or career_weights.sum() <= 0 or macro_weights.sum() <= 0 or len(macro_weights) != len(MACROS):
        raise ValueError("Forecast weights must be nonnegative, nonzero, and match the macro categories")
    probabilities = np.outer(career_weights/career_weights.sum(), macro_weights/macro_weights.sum()).ravel()  # (c*6,)
    rng = np.random.default_rng(config.seed + 7919)
    assignments = rng.choice(len(probabilities), size=config.forecast_paths, p=probabilities)  # (forecast_paths,)
    samples, groups = [], []
    for group in np.unique(assignments):
        n = int(np.sum(assignments == group))
        career, macro = careers[group//len(MACROS)], list(MACROS)[group % len(MACROS)]
        location = "bay_area" if career == "profluent" else "newark" if career == "faculty_ud" else "squirrel_hill"
        scenario = Scenario(career=career, career_year=2027 if career in ("ud", "ud_blend") else 2028, location=location, move_year=2028 if career in ("profluent", "faculty_ud") else 2029,
                            marriage_year=2029, births=(2030, 2032), housing="rent", macro=macro,
                            benefits="retained", label="Forecast conditional on marriage 2029, children 2030/2032, Amanda working")
        local = dataclasses.replace(config, paths=max(n, 2), seed=config.seed + 104729 * (int(group)+1))
        draws = economic_paths(local, macro)  # (max(n,2),t,8)
        household, business = simulate(local, scenario, draws, forecast=True)
        summary, terminal, bands = summarize(SimulationResult(household, business), local, scenario)
        terminal = terminal.iloc[:n].copy()
        terminal["career"] = career
        terminal["macro"] = macro
        terminal["weight"] = 1 / config.forecast_paths
        samples.append(terminal)
        groups.append({"career": career, "macro": macro, "prior_probability": float(probabilities[group]), "sampled_paths": n})
    combined = pd.concat(samples, ignore_index=True)
    combined.to_parquet(folder / "forecast.parquet", index=False)
    write_json(folder / "forecast_assumptions.json", {
        "kind": "subjective_probability_forecast", "paths": config.forecast_paths,
        "fixed_choices": "Marriage 2029; births 2030 and 2032; Amanda continues working; renting; retained UD benefits conditional on eligibility",
        "caveat": "Not a probability distribution over the exhaustive grid. Career and macro priors are subjective; grant outcomes/delays are sampled.",
        "darpa_award_prior": next((g.award_probability for g in config.business.grants if g.name == "darpa"), None), "groups": groups,
        "median_net_worth": float(combined.net_worth.median()),
        "p025_net_worth": float(combined.net_worth.quantile(.025)),
        "p975_net_worth": float(combined.net_worth.quantile(.975)),
        "shortfall_probability": float(combined.ever_shortfall.mean()),
    })
    return combined
