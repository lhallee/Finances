"""Translate dated scenario events into numerical monthly inputs."""

import numpy as np

from collections import namedtuple
from dataclasses import replace

from .configuration import RunConfig
from .economics import monthly_dates
from .reference import AMANDA_FACTORS, CAREER_SALARY, LOCATIONS, STATES
from .scenarios import Scenario


SCHEDULE_COLUMNS = ("year", "month", "state", "local_tax", "cost_factor", "rent", "home_price",
                    "property_tax_rate", "childcare_price", "children", "new_birth", "married",
                    "wedding", "amanda_stopped", "amanda_leave", "logan_wage", "ud_wage",
                    "founder_fte", "logan_benefits", "logan_match", "amanda_salary", "move",
                    "purchase_requested", "exit", "delaware_source", "children_under13", "darpa_blend", "fallback_wage", "fallback_match",
                    "logan_new_role", "logan_raise_due", "amanda_raise_due")

SCHEDULE = namedtuple("ScheduleColumns", SCHEDULE_COLUMNS)(*range(len(SCHEDULE_COLUMNS)))


def build_schedule(config: RunConfig, scenario: Scenario) -> np.ndarray:
    dates = monthly_dates(config)
    schedule = np.zeros((len(dates), len(SCHEDULE_COLUMNS)), dtype=float)  # (t, 26)
    event_month = config.grid.event_month
    career_month = scenario.career_month or event_month
    if not 1 <= career_month <= 12 or scenario.prior_career not in ("ud", "ud_blend"):
        raise ValueError("Invalid career transition month or prior career")
    h = config.household
    for m, date in enumerate(dates):
        year, month = date.year, date.month
        switched = (year, month) >= (scenario.career_year, career_month)
        moved = scenario.location != "parents" and (year, month) >= (scenario.move_year, event_month)
        location_name = scenario.location if moved else "parents"
        location = replace(LOCATIONS[location_name], **config.location_overrides.get(location_name, {}))
        salaries = {**CAREER_SALARY, **config.career_salary_overrides}
        births = [(birth, event_month) for birth in scenario.births if (year, month) >= (birth, event_month)]
        count = len(births)
        stopped = scenario.stop_after_birth > 0 and count >= scenario.stop_after_birth
        new_birth = (year, month) in [(birth, event_month) for birth in scenario.births]
        leave = any(0 <= (year - birth) * 12 + month - event_month < h.parental_leave_months for birth in scenario.births) and not stopped
        married = scenario.marriage_year > 0 and (year, month) >= (scenario.marriage_year, event_month)
        wage, ud, founder, benefits, match = h.logan.salary, h.logan.salary, 0., 1., h.logan.employer_match
        career = scenario.career if switched else scenario.prior_career
        if career in ("ud_blend", "faculty_blend") and (year, month) >= (2027, 2):
            ud = h.logan.salary * .45 if career == "ud_blend" else salaries[career] * .45
            wage, founder = ud, .55
            benefits = float(scenario.benefits == "retained")
            match = h.logan.employer_match if benefits else 0.
        elif career == "ud_nonrenewal" and (year, month) >= (2028, 9):
            wage, ud, benefits, match = 0., 0., 0., 0.
        elif career == "synthyra":
            wage, ud, founder, benefits, match = 0., 0., 1., 1., .04
        elif career not in ("ud", "ud_blend", "faculty_blend", "ud_nonrenewal"):
            wage, ud, benefits, match = salaries[career], 0., 1., .05
            if career.startswith("faculty"):
                # Salary anchor is a nine-month academic salary; two summer months at 50% support.
                wage *= (9 + config.faculty_summer_months * config.faculty_summer_support_fraction) / 9
            if career == "industry_blend":
                founder = .25
        if scenario.grant_case == "failed" and career == "ud_blend":
            wage, ud, founder, benefits, match = h.logan.salary, h.logan.salary, 0., 1., h.logan.employer_match
        amanda_salary = h.amanda.salary * AMANDA_FACTORS[scenario.amanda_career] * (location.amanda_factor if moved else 1)
        care_units = sum(1. if (year-birth)*12+month-event_month < 60 else config.school_age_childcare_fraction for birth, _ in births)
        care_per_child = location.childcare * care_units / count if count else location.childcare
        has_new_role = scenario.career not in ("ud", "ud_blend", "ud_nonrenewal") and switched
        role_months = (year - scenario.career_year) * 12 + month - career_month if has_new_role else m
        logan_raise = role_months > 0 and role_months % 12 == 0
        schedule[m] = (year, month, STATES[location.state], location.local_tax, location.cost_factor,
                       location.rent, location.home_price, location.property_tax, care_per_child,
                       count, float(new_birth), float(married), float((year, month) == (scenario.marriage_year, event_month)),
                       float(stopped), float(leave), wage / 12, ud / 12, founder, benefits, match,
                       amanda_salary / 12, float(moved and (year, month) == (scenario.move_year, event_month)),
                       float(moved and scenario.housing == "buy"),
                       float(scenario.exit_value > 0 and (year, month) == (scenario.exit_year, event_month)),
                       ud / 12 * config.taxes.ud_delaware_source_share, count,
                       float(career == "ud_blend" and "darpa" in scenario.grants), h.logan.salary / 12, h.logan.employer_match,
                       float(has_new_role and role_months == 0), float(logan_raise), float(m > 0 and m % 12 == 0))
    return schedule  # (t, 26)
