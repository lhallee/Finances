"""Readable configured events for a clicked scenario, separate from realized paths."""

import pandas as pd

from display_names import display_name, option_name
from finance_sim.configuration import RunConfig
from finance_sim.scenarios import Scenario


def event_timeline(scenario: Scenario, config: RunConfig) -> pd.DataFrame:
    month = config.grid.event_month
    events = []

    def add(year: int, event: str, event_month: int = month) -> None:
        events.append({"Date": f"{year:04d}-{event_month:02d}", "Configured Event": event})

    if scenario.career == "ud_blend" or scenario.prior_career == "ud_blend":
        add(2027, "Proposed UD + Synthyra split, conditional on funding; February start", 2)
    if scenario.career == "ud_nonrenewal":
        add(2028, "UD appointment ends without renewal", 9)
    elif scenario.career not in ("ud", "ud_blend"):
        add(scenario.career_year, f"Start {option_name('career', scenario.career)}", scenario.career_month or month)
    if scenario.location != "parents":
        add(scenario.move_year, f"Move to {option_name('location', scenario.location)}; {option_name('housing', scenario.housing)}")
    if scenario.marriage_year:
        add(scenario.marriage_year, "Get married")
    for number, year in enumerate(scenario.births, 1):
        add(year, f"Child {number} is born")
        if scenario.stop_after_birth == number:
            add(year, "Amanda stops paid work; paid childcare ends for all children; health coverage changes")
    for grant in config.business.grants:
        if grant.name in scenario.grants:
            events.append({"Date": grant.start[:7], "Configured Event": f"{option_name('grants', grant.name)}: {option_name('grant_case', scenario.grant_case)} case; receipts depend on spending and delays"})
    if scenario.company_outcome == "vc":
        for funding in config.business.rounds:
            add(funding.year, f"Financing round: ${funding.amount/1e6:g} million ({display_name(funding.kind)})", 1)
    if scenario.exit_value:
        add(scenario.exit_year, f"{option_name('exit_type', scenario.exit_type)} at {option_name('exit_value', scenario.exit_value)} enterprise value")
    return pd.DataFrame(events, columns=["Date", "Configured Event"]).sort_values("Date", kind="stable")
