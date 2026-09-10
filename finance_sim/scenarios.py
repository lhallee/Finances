"""Lazy exhaustive scenario grids and explicitly smaller named presets."""

from __future__ import annotations

import hashlib
import itertools
import json

from dataclasses import asdict, dataclass, replace
from collections.abc import Iterator

from .configuration import RunConfig
from .reference import AMANDA_FACTORS, CAREER_SALARY, LOCATIONS, MACROS


@dataclass(frozen=True)
class Scenario:
    career: str = "ud_blend"
    career_year: int = 2027
    location: str = "parents"
    move_year: int = 2029
    housing: str = "free"
    marriage_year: int = 0
    births: tuple[int, ...] = ()
    stop_after_birth: int = 0
    amanda_career: str = "pharma"
    company_mode: str = "business"
    company_outcome: str = "saas"
    macro: str = "baseline"
    benefits: str = "retained"
    grants: tuple[str, ...] = ("darpa",)
    grant_case: str = "awarded"
    exit_value: float = 0
    exit_year: int = 0
    exit_type: str = "equity"
    label: str = ""
    prior_career: str = "ud"
    career_month: int = 0  # Zero uses the shared event month.

    @property
    def id(self) -> str:
        raw = asdict(self)
        raw.pop("label")
        # Preserve identifiers for existing scenarios with the original defaults.
        if raw["prior_career"] == "ud":
            raw.pop("prior_career")
        if raw["career_month"] == 0:
            raw.pop("career_month")
        return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:20]

    def record(self) -> dict:
        return {"scenario_id": self.id, **asdict(self)}


def family_choices(config: RunConfig) -> list[tuple[tuple[int, ...], int]]:
    choices = []
    for count in range(config.grid.maximum_children + 1):
        for births in itertools.combinations(config.grid.years, count):
            if any((b - a) * 12 < config.grid.minimum_birth_spacing_months for a, b in zip(births, births[1:])):
                continue
            choices.extend((births, stop) for stop in range(count + 1))
    return choices


def compatible(career: str, location: str) -> bool:
    """Whether the location matches the role's reference geography, not a gate."""
    required = {"biohub_engineer": ("nyc", "bay_area"), "biohub_scientist": ("nyc", "bay_area"),
                "profluent": ("bay_area",), "merck": ("parents", "philadelphia"),
                "pharma_agritech_pi": ("parents", "philadelphia", "newark", "boston", "bay_area", "nyc"),
                "pharma_agritech_director": ("parents", "philadelphia", "newark", "boston", "bay_area", "nyc"),
                "bio_ai_engineer": ("nyc", "bay_area", "boston"), "bio_ai_research": ("nyc", "bay_area", "boston"),
                "protein_design": ("nyc", "bay_area", "boston"),
                "faculty_ud": ("parents", "newark", "philadelphia"),
                "faculty_ca": ("bay_area", "san_diego"), "faculty_ma": ("boston", "amherst"),
                "industry_blend": ("bay_area", "nyc"), "faculty_blend": ("parents", "newark")}
    return career not in required or location in required[career]


def validate_grid(config: RunConfig) -> None:
    for selected, supported, label in ((config.grid.careers, CAREER_SALARY, "career"),
                                     (config.grid.locations, LOCATIONS, "location"),
                                     (config.grid.macros, MACROS, "macro"),
                                     (config.grid.amanda_careers, AMANDA_FACTORS, "Amanda career")):
        if not selected or set(selected) - set(supported):
            raise ValueError(f"Unknown or empty {label} selection: {selected}")
    names = {g.name for g in config.business.grants}
    if any(set(p) - names for p in config.grid.funding_portfolios):
        raise ValueError("Funding portfolio references an undefined grant")


def count_scenarios(config: RunConfig, preset: str = "standard") -> int:
    validate_grid(config)
    if preset == "standard":
        return sum(1 for _ in standard_bases(config)) * len(practical_families(config)) * len(set(config.grid.practical_marriage_years)) * len(set(config.grid.macros))
    if preset != "full":
        return sum(1 for _ in generate_scenarios(config, preset))
    grid = config.grid
    n_years = len(grid.years)
    career_locations = 0
    for career in grid.careers:
        dates = 1 if career in ("ud", "ud_blend", "ud_nonrenewal") else n_years
        benefits = len(grid.benefits) if career in ("ud_blend", "faculty_blend") else 1
        for location in grid.locations:
            if grid.independent_career_locations or compatible(career, location):
                # Career and relocation dates are independent.
                moves = 1 if location == "parents" else n_years
                housing = 1 if location == "parents" else len(grid.housing)
                career_locations += dates * benefits * moves * housing
    portfolios = sum(1 if not p else len(grid.grant_cases) for p in grid.funding_portfolios)
    exits = sum(1 if v == 0 else n_years * len(grid.exit_types) for v in grid.exit_values)
    company = sum(1 if outcome in ("dormant", "failure") else exits for outcome in grid.company_outcomes)
    return career_locations * (n_years + 1) * len(family_choices(config)) * len(grid.amanda_careers) * len(grid.company_modes) * len(grid.macros) * portfolios * company


def practical_families(config: RunConfig) -> list:
    families = []
    for births in dict.fromkeys(map(tuple, config.grid.practical_birth_schedules)):
        if len(births) > config.grid.maximum_children or any(y not in config.grid.years for y in births):
            raise ValueError(f"Invalid practical birth schedule: {births}")
        if any((b-a)*12 < config.grid.minimum_birth_spacing_months for a,b in zip(births, births[1:])):
            raise ValueError(f"Invalid practical birth spacing: {births}")
        families.extend((births, stop) for stop in range(len(births) + 1))
    return families


def standard_bases(config: RunConfig) -> Iterator[Scenario]:
    """Deduplicate inactive/base dimensions before expanding the family grid."""
    seen = set()
    years = config.grid.practical_transition_years
    if not years or any(year not in config.grid.years for year in years):
        raise ValueError("Practical transition years must be within grid.years")
    for name, settings in config.grid.base_schemas.items():
        base = scenario_from_record(settings)
        locations = config.grid.locations if config.grid.independent_career_locations else (base.location,)
        for location in locations:
            if not config.grid.independent_career_locations and not compatible(base.career, location):
                raise ValueError(f"Incompatible career/location in schema {name}")
            homes = (("free",) if location == "parents" else config.grid.housing) if config.grid.independent_career_locations else (base.housing,)
            benefits = config.grid.benefits if base.career in ("ud_blend", "faculty_blend") else (base.benefits,)
            career_years = years if base.career not in ("ud", "ud_blend", "ud_nonrenewal") else (base.career_year,)
            moves = years if location != "parents" else (2029,)
            for home, benefit, career_year, move in itertools.product(homes, benefits, career_years, moves):
                candidate = replace(base, location=location, housing=home, benefits=benefit, career_year=career_year,
                                    move_year=move, births=(), stop_after_birth=0, marriage_year=0, macro="baseline", label="")
                if candidate not in seen:
                    seen.add(candidate)
                    yield replace(candidate, label=name)


def generate_scenarios(config: RunConfig, preset: str = "standard") -> Iterator[Scenario]:
    if preset == "standard":
        families = practical_families(config)
        for base in standard_bases(config):
            for (births, stop), marriage, macro in itertools.product(
                families, dict.fromkeys(config.grid.practical_marriage_years), dict.fromkeys(config.grid.macros)):
                yield replace(base, births=births, stop_after_birth=stop, marriage_year=marriage, macro=macro)
        return
    if preset == "baseline":
        yield Scenario(label="Current household and DARPA")
        return
    if preset in ("demo", "family"):
        locations = ("parents", "squirrel_hill", "norway") if preset == "demo" else ("squirrel_hill",)
        for location, count, benefits in itertools.product(locations, range(4), ("retained", "lost")):
            births = tuple(2029 + 2 * i for i in range(count))
            for stop in range(count + 1):
                yield Scenario(location=location, housing="free" if location == "parents" else "rent",
                               marriage_year=2028, births=births, stop_after_birth=stop,
                               benefits=benefits, label=f"{location}, {count} children, stop {stop}, {benefits}")
        return
    if preset != "full":
        raise ValueError(f"Unknown preset: {preset}")
    validate_grid(config)
    grid = config.grid
    families = family_choices(config)
    for career, location in itertools.product(grid.careers, grid.locations):
        if not grid.independent_career_locations and not compatible(career, location):
            continue
        career_years = (2027,) if career in ("ud", "ud_blend", "ud_nonrenewal") else grid.years
        benefits = grid.benefits if career in ("ud_blend", "faculty_blend") else ("retained",)
        for career_year in career_years:
            moves = (career_year,) if location == "parents" else grid.years
            homes = ("free",) if location == "parents" else grid.housing
            for move, home, benefit, marriage, family, amanda, mode, macro, grants, outcome in itertools.product(
                moves, homes, benefits, (0, *grid.years), families, grid.amanda_careers,
                grid.company_modes, grid.macros, grid.funding_portfolios, grid.company_outcomes):
                grant_cases = grid.grant_cases if grants else ("awarded",)
                exits = (0,) if outcome in ("dormant", "failure") else grid.exit_values
                for grant_case, value in itertools.product(grant_cases, exits):
                    for exit_year, exit_type in itertools.product(grid.years if value else (0,), grid.exit_types if value else ("equity",)):
                        yield Scenario(career, career_year, location, move, home, marriage,
                                       family[0], family[1], amanda, mode, outcome, macro,
                                       benefit, tuple(grants), grant_case, value, exit_year, exit_type)


def scenario_from_record(record: dict) -> Scenario:
    fields = Scenario.__dataclass_fields__
    values = {key: value for key, value in record.items() if key in fields}
    for key in ("births", "grants"):
        if key in values:
            if isinstance(values[key], str):
                values[key] = tuple(int(v) for v in values[key].split(",") if v) if key == "births" else tuple(v for v in values[key].split(",") if v)
            else:
                values[key] = tuple(int(v) for v in values[key]) if key == "births" else tuple(values[key])
    return Scenario(**values)
