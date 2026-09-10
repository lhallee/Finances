"""Reject editable input mistakes before they silently change grid coverage."""

import pytest

from finance_sim.configuration import RunConfig
from finance_sim.scenarios import count_scenarios, generate_scenarios, scenario_from_record


@pytest.mark.parametrize("settings, message", [
    ({"carer": "bio_ai_engineer"}, "Unknown fields"),
    ({"career": "bio_ai_enginer"}, "Unknown career"),
    ({"grants": ["undefined_grant"]}, "Undefined grant"),
    ({"grants": ["darpa", "darpa"]}, "Duplicate grants"),
    ({"housing": "mortage"}, "Unknown housing"),
])
def test_practical_schema_errors_fail_before_counting(settings, message):
    config = RunConfig()
    config.grid.base_schemas = {"edited": settings}
    with pytest.raises(ValueError, match=message):
        count_scenarios(config)


@pytest.mark.parametrize("axis, values", [
    ("years", (2027, 2027)),
    ("careers", ("ud", "ud")),
    ("funding_portfolios", ((), ())),
    ("practical_birth_schedules", ((2029,), (2029,))),
])
def test_duplicate_grid_choices_rejected(axis, values):
    config = RunConfig()
    setattr(config.grid, axis, values)
    with pytest.raises(ValueError, match=f"Duplicate choices in grid.{axis}"):
        config.validate()
    with pytest.raises(ValueError, match="Duplicate choices"):
        count_scenarios(config, "full")


def test_unsorted_birth_calendar_rejected():
    config = RunConfig()
    config.grid.years = (2028, 2027)
    with pytest.raises(ValueError, match="chronological"):
        config.validate()


def test_saved_summary_parser_still_accepts_derived_columns():
    scenario = scenario_from_record({"career": "bio_ai_engineer", "net_worth_p50": 100_000})
    assert scenario.career == "bio_ai_engineer"


def test_valid_practical_grid_count_and_identifiers_remain_equal():
    config = RunConfig()
    config.grid.base_schemas = {"edited": {"career": "bio_ai_engineer"}}
    config.grid.locations = ("parents", "nyc")
    config.grid.macros = ("baseline",)
    rows = list(generate_scenarios(config))
    assert len(rows) == count_scenarios(config) == len({row.id for row in rows})
