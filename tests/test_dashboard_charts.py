"""The overview retains every selected scenario and maps point clicks to IDs."""

import numpy as np
import pandas as pd

from dashboard_charts import all_scenario_chart, clicked_scenario
from display_names import display_name, option_name
from finance_sim.configuration import RunConfig
from finance_sim.scenarios import Scenario, generate_scenarios
from finance_sim.schedule import SCHEDULE, build_schedule
from scenario_details import event_timeline


def test_chart_filter_and_click_mapping():
    rows = pd.DataFrame({"scenario_id": ["a", "a", "b", "b", "c", "c"],
                         "date": pd.to_datetime(["2027-01-01", "2027-02-01"] * 3), "q500": [1, 2, 3, 4, 5, 6]})
    selection = pd.DataFrame({"scenario_id": ["a", "b", "c"], "career": ["ud", "ud", "biohub_engineer"]})
    chart = all_scenario_chart(rows, selection)
    assert chart.layout.meta["scenario_count"] == 3
    assert sum(np.isfinite(trace.y).sum() for trace in chart.data) == 6
    assert any(trace.name == "Bio-AI / Protein Design" for trace in chart.data)
    assert np.isnan(chart.data[0].y[2])
    event = {"selection": {"points": [{"curve_number": 0, "point_index": 4}]}}
    assert clicked_scenario(event, chart) == "b"
    filtered = all_scenario_chart(rows, selection[selection.career != "ud"])
    assert filtered.layout.meta["scenario_count"] == 1
    assert filtered.data[0].meta["scenario_ids"] == ["c"]


def test_presentation_names():
    assert display_name("biohub_engineer") == "Biohub Engineer"
    assert display_name("ud_blend") == "UD + Synthyra"
    assert option_name("stop_after_birth", 0) == "Continue Working"
    assert option_name("exit_value", 0) == "No Exit"


def test_signed_axis_and_uncertainty_envelope_keep_every_extreme():
    rows = pd.DataFrame({"scenario_id": ["a", "a", "b", "b"],
                         "date": pd.to_datetime(["2027-01-01", "2027-02-01"] * 2),
                         "q500": [-50_000, -20_000, 1e6, 1e8],
                         "q025": [-100_000, -50_000, 2e5, 5e7], "q975": [0, 5e4, 2e6, 2e8]})
    selected = pd.DataFrame({"scenario_id": ["a", "b"], "career": ["ud", "profluent"]})
    for scale in ("Balanced", "Linear"):
        chart = all_scenario_chart(rows, selected, "95%", scale)
        low, high = chart.layout.yaxis.range
        assert low < min(np.nanmin(t.y) for t in chart.data)
        assert high > max(np.nanmax(t.y) for t in chart.data)
        assert low < 0 < high
        if scale == "Balanced":
            assert any("−" in tick for tick in chart.layout.yaxis.ticktext)
            np.testing.assert_allclose(chart.data[0].customdata[:2], [-50_000, -20_000])
        assert clicked_scenario({"selection": {"points": [{"curve_number": 3, "point_index": 0}]}}, chart) is None


def test_only_eight_locations_in_new_presets():
    config = RunConfig()
    wanted = {"parents", "squirrel_hill", "norway", "newark", "philadelphia", "nyc", "bay_area", "boston"}
    assert set(config.grid.locations) == wanted
    assert {s.location for s in generate_scenarios(config)} == wanted
    assert {s.location for s in generate_scenarios(config, "demo")} <= wanted


def test_practical_2027_transitions_and_known_split():
    config = RunConfig()
    config.grid.base_schemas = {"new_job": {"career": "profluent", "location": "bay_area", "housing": "rent"},
                                "remote_move": {"location": "squirrel_hill", "housing": "rent"}}
    config.grid.practical_birth_schedules = ((),)
    config.grid.practical_marriage_years = (0,)
    config.grid.macros = ("baseline",)
    cases = list(generate_scenarios(config))
    careers = [s for s in cases if s.career == "profluent"]
    assert {s.career_year for s in careers} == {2027, 2029, 2032}
    assert {(s.career_year, s.move_year) for s in careers if s.location != "parents"} == {(c, m) for c in (2027, 2029, 2032) for m in (2027, 2029, 2032)}
    early = next(s for s in careers if s.career_year == 2027 and s.move_year == 2027 and s.location != "parents")
    schedule = build_schedule(config, early)
    assert schedule[2, SCHEDULE.ud_wage] > 0
    assert schedule[3, SCHEDULE.ud_wage] == 0
    assert schedule[3, SCHEDULE.move] == 1
    blend = next(s for s in cases if s.career == "ud_blend" and s.move_year == 2027)
    schedule = build_schedule(config, blend)
    assert schedule[3, SCHEDULE.founder_fte] == 0
    assert schedule[4, SCHEDULE.founder_fte] == .55


def test_scenario_event_timeline():
    scenario = Scenario(career="profluent", career_year=2027, location="bay_area", move_year=2027,
                        births=(2029, 2031), stop_after_birth=2, marriage_year=2028)
    timeline = event_timeline(scenario, RunConfig())
    assert any("Start Profluent" in value for value in timeline["Configured Event"])
    stop = timeline[timeline["Configured Event"].str.contains("Amanda stops")]
    assert stop.iloc[0].Date == "2031-01"
    financed = event_timeline(Scenario(company_outcome="vc"), RunConfig())
    assert any("(Priced)" in value for value in financed["Configured Event"])
