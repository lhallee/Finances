"""Typed inputs: 2026 spending/salary anchors, nominal fixed contracts and balances."""

from __future__ import annotations

import dataclasses
import importlib.util
import json

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Person:
    name: str = "Logan"
    age: int = 27
    salary: float = 70_000
    cash: float = 5000
    brokerage: float = 0
    brokerage_basis: float = 0
    retirement: float = 0
    monthly_personal_spend: float = 1500
    retirement_rate: float = .05
    employer_match: float = .11
    medical_monthly: float = 150


@dataclass
class Loan:
    name: str
    balance: float
    apr: float
    minimum: float
    deferred_until: str = "2026-01-01"
    subsidized: bool = False
    repayment_months: int = 120


@dataclass
class Household:
    logan: Person = field(default_factory=Person)
    amanda: Person = field(default_factory=lambda: Person(
        name="Amanda", age=28, salary=58_000, cash=10_000,
        retirement=12_000, monthly_personal_spend=1100,
        retirement_rate=.05, employer_match=.04, medical_monthly=150))
    # Real balances belong in ignored inputs/household.json, not library defaults.
    loans: list[Loan] = field(default_factory=list)
    cash_target_monthly: float = 2000
    cash_cap: float | None = None
    cash_apy: float = .035
    debt_hurdle: float = .0658
    debt_transition_band: float = .02
    blog_subscribers: float = 1
    blog_net_per_subscriber: float = 6.61
    blog_monthly_growth: float = 0
    annual_raise: float = .03
    ud_job_loss_annual_probability: float = 0
    family_medical_monthly: float = 650
    replacement_adult_medical: float = 600
    replacement_child_medical: float = 250
    domestic_partner_eligible: bool = False
    medical_annual_oop: float = 1200
    child_monthly_other: float = 450
    birth_cost: float = 4500
    wedding_cost: float = 20_000
    parental_leave_months: int = 3
    parental_leave_pay_fraction: float = .5
    amanda_retirement_enabled: bool = True
    job_loss_annual_probability: float = .025
    job_gap_months: int = 4
    repair_annual_probability: float = .12
    repair_cost: float = 2500
    logan_2026_prior_wages: float = 23_333.33
    amanda_2026_prior_wages: float = 43_500
    prior_2026_withholding: float | None = None


@dataclass
class Housing:
    down_payment: float = .20
    closing_cost: float = .03
    sale_cost: float = .06
    mortgage_rate: float = .0625
    mortgage_years: int = 30
    max_dti: float = .36
    reserve_months: float = 6
    maintenance_rate: float = .015
    insurance_rate: float = .005
    pmi_rate: float = .006
    moving_cost: float = 6000
    bedroom_premium_per_child: float = .08


@dataclass
class Grant:
    name: str = "darpa"
    amount: float = 2_000_000
    start: str = "2027-02-01"
    months: int = 24
    award_probability: float = .95
    receipt_delay_months: int = 0
    delay_probability: float = .15
    delayed_months: int = 3
    effort_fraction: float = .55
    cost_share: float = 0
    overhead_rate: float = .20
    direct_cost_monthly: float = 20_000
    follow_on_probability: float = .20
    follow_on_annual_revenue: float = 120_000
    follow_on_months: int = 24
    equity_fraction: float = 0
    # (month offset, fraction of total award); empty means cost reimbursement.
    milestones: tuple[tuple[int, float], ...] = ()


@dataclass
class FundingRound:
    year: int = 2028
    amount: float = 10_000_000
    pre_money: float = 30_000_000
    pool_topup: float = .10
    preference: float = 1
    participating: bool = False
    kind: str = "priced"
    conversion_year: int = 2029
    valuation_cap: float = 20_000_000
    discount: float = .20
    interest: float = .08


@dataclass
class Business:
    cash: float = 10_000
    monthly_revenue: float = 0
    monthly_overhead: float = 1000
    annual_raise: float = .03
    starting_clients: float = 0
    monthly_new_clients: float = 1
    monthly_churn: float = .025
    annual_revenue_per_client: float = 12_000
    gross_margin: float = .70
    licensing_annual_probability: float = .15
    licensing_upfront: float = 500_000
    licensing_annual_royalty: float = 150_000
    payroll_load: float = .18
    initial_staff_fte: float = 1
    staff_annual_cost: float = 120_000
    max_staff_fte: float = 12
    founder_fulltime_salary: float = 150_000
    vc_founder_salary: float = 250_000
    revenue_salary_fraction: float = .12
    salary_cap: float = 500_000
    distributions_fraction: float = .10
    logan_equity: float = .35
    braydon_issuance: float = .10
    diane_issuance: float = .001
    stephen_participation: float = .005
    stephen_grant_date: str = "2026-04-27"
    stephen_expiration_years: int = 7
    equity_basis: float = 0
    tax_classification: str = "partnership"
    exit_transaction_cost: float = .04
    exit_escrow_fraction: float = .10
    exit_escrow_months: int = 12
    exit_earnout_fraction: float = 0
    exit_earnout_months: int = 24
    exit_earnout_probability: float = .8
    exit_debt: float = 0
    asset_basis: float = 0
    qsbs_eligible: bool = False
    qsbs_conversion_value: float = 0
    llc_hot_asset_fraction: float = 0
    fulltime_founder_effort: float = 1
    rounds: list[FundingRound] = field(default_factory=lambda: [FundingRound(), FundingRound(2030, 25_000_000, 100_000_000, .05)])
    grants: list[Grant] = field(default_factory=lambda: [
        Grant(),
        Grant("sbir_phase1", 300_000, "2028-01-01", 12, .25,
              effort_fraction=.2, direct_cost_monthly=8000,
              follow_on_probability=.35, follow_on_annual_revenue=500_000),
        Grant("foundation", 750_000, "2028-01-01", 24, .15,
              effort_fraction=.2, direct_cost_monthly=12_000),
        Grant("pharma_agritech", 500_000, "2028-01-01", 18, .25,
              effort_fraction=.2, direct_cost_monthly=10_000,
              follow_on_probability=.4, follow_on_annual_revenue=300_000),
        Grant("ud_contract", 100_000, "2027-02-01", 12, .5,
              effort_fraction=.1, direct_cost_monthly=2500),
    ])


@dataclass
class Economics:
    annual_return: float = .08
    annual_volatility: float = .20
    annual_inflation: float = .025
    annual_home_growth: float = .03
    dividend_yield: float = .008
    return_mode: str = "parametric"
    history_csv: str | None = None
    block_months: int = 12
    # Monthly aligned columns: equity_return, inflation, home_return, cash_rate.
    reference_date: str = "2026-09-10"


@dataclass
class TaxPolicy:
    base_year: int = 2026
    index_rate: float = .025
    employee_deferral_limit: float = 24_500
    social_security_base: float = 184_500
    federal_standard_single: float = 16_100
    child_credit: float = 2200
    ud_delaware_source_share: float = 1
    qsbs_state_conformity: tuple[str, ...] = ("DE", "ME", "NY", "MA")
    # State/local rules are planning approximations; see sources.json and MODEL.md.


@dataclass
class RetirementPolicy:
    # Household stops paid work together. All dollar assumptions are October 2026 USD.
    life_expectancy: int = 100
    maximum_retirement_age: int = 80
    success_target: float = .95
    withdrawal_rate: float = .04
    annual_spending_floor: float = 50_000
    annual_spending_override: float | None = None
    annual_healthcare: float = 18_000
    withdrawal_tax_rate: float = .20
    account_access_age: float = 59.5
    real_return: float = .03
    annual_volatility: float = .15
    retirement_paths: int = 4096
    project_beyond_horizon: bool = True
    child_support_end_age: int = 22
    college_start_age: int = 18
    college_annual_cost: float = 28_500  # Editable 2026-dollar public in-state total-cost proxy.
    college_parent_share: float = .5
    future_inflation: float = .025  # Deflates fixed nominal mortgage payments after 2036.


@dataclass
class Grid:
    # The practical preset crosses these explicit base cases with every declared
    # marriage, birth/working policy and macro below. Full uses the separate axes.
    base_schemas: dict[str, dict] = field(default_factory=lambda: {
        "parents_darpa": {},
        "pittsburgh_rent": {"location": "squirrel_hill", "housing": "rent"},
        "pittsburgh_buy": {"location": "squirrel_hill", "housing": "buy"},
        "norway_rent": {"location": "norway", "housing": "rent"},
        "norway_buy": {"location": "norway", "housing": "buy"},
        "ud_continues": {"career": "ud"},
        "ud_no_synthyra_income": {"career": "ud", "company_outcome": "dormant", "grants": []},
        "industry_no_synthyra_income": {"career": "bio_ai_engineer", "company_outcome": "dormant", "grants": []},
        "bio_ai_research": {"career": "bio_ai_research", "career_year": 2028},
        "bio_ai_engineering": {"career": "bio_ai_engineer", "career_year": 2028},
        "protein_design": {"career": "protein_design", "career_year": 2028},
        "pharma_pi_philly": {"career": "pharma_agritech_pi", "location": "philadelphia", "housing": "buy"},
        "pharma_pi_boston": {"career": "pharma_agritech_pi", "location": "boston", "housing": "rent"},
        "pharma_pi_bay": {"career": "pharma_agritech_pi", "location": "bay_area", "housing": "rent"},
        "pharma_director_philly": {"career": "pharma_agritech_director", "location": "philadelphia", "housing": "buy"},
        "pharma_director_boston": {"career": "pharma_agritech_director", "location": "boston", "housing": "rent"},
        "pharma_director_bay": {"career": "pharma_agritech_director", "location": "bay_area", "housing": "rent"},
        "faculty_ud": {"career": "faculty_ud", "career_year": 2029, "location": "newark", "housing": "buy"},
        "faculty_ca": {"career": "faculty_ca", "career_year": 2029, "location": "bay_area", "housing": "rent"},
        "faculty_boston": {"career": "faculty_ma", "career_year": 2029, "location": "boston", "housing": "rent"},
        "faculty_boston_buy": {"career": "faculty_ma", "career_year": 2029, "location": "boston", "housing": "buy"},
        "industry_concurrent": {"career": "industry_blend", "career_year": 2028, "move_year": 2028, "location": "bay_area", "housing": "rent"},
        "faculty_concurrent": {"career": "faculty_blend", "career_year": 2029},
        "synthyra_vc": {"career": "synthyra", "career_year": 2028, "company_outcome": "vc"},
        "licensing": {"company_outcome": "licensing"},
        "hybrid": {"company_outcome": "hybrid"},
        "company_failure": {"company_outcome": "failure"},
        "darpa_delayed": {"grant_case": "delayed"},
        "darpa_reduced": {"grant_case": "reduced"},
        "darpa_failed": {"grant_case": "failed"},
        "sbir_sequence": {"grants": ["darpa", "sbir_phase1"]},
        "foundation": {"grants": ["darpa", "foundation"]},
        "joint_research": {"grants": ["darpa", "pharma_agritech"]},
        "ud_contract": {"grants": ["darpa", "ud_contract"]},
        "stacked_funding": {"grants": ["darpa", "sbir_phase1", "foundation", "pharma_agritech", "ud_contract"]},
        "bootstrap_exit_10m": {"exit_value": 10e6, "exit_year": 2032},
        "vc_exit_10m": {"company_outcome": "vc", "exit_value": 10e6, "exit_year": 2032},
        "vc_exit_100m": {"company_outcome": "vc", "exit_value": 100e6, "exit_year": 2032},
        "vc_exit_1b": {"company_outcome": "vc", "exit_value": 1e9, "exit_year": 2035},
        "asset_exit_100m": {"company_outcome": "vc", "exit_value": 100e6, "exit_year": 2032, "exit_type": "asset"},
    })
    practical_birth_schedules: tuple[tuple[int, ...], ...] = ((), (2029,), (2029, 2031), (2029, 2031, 2033))
    practical_marriage_years: tuple[int, ...] = (0, 2028)
    # Coherent early, middle and later transitions. Use grid.years for every year.
    practical_transition_years: tuple[int, ...] = (2027, 2029, 2032)
    independent_career_locations: bool = True
    years: tuple[int, ...] = tuple(range(2027, 2037))
    event_month: int = 1
    careers: tuple[str, ...] = ("ud", "ud_blend", "synthyra", "bio_ai_engineer", "bio_ai_research", "protein_design", "pharma_agritech_pi", "pharma_agritech_director", "faculty_ud", "faculty_ca", "faculty_ma", "industry_blend", "faculty_blend")
    locations: tuple[str, ...] = ("parents", "squirrel_hill", "norway", "newark", "philadelphia", "nyc", "bay_area", "boston")
    housing: tuple[str, ...] = ("rent", "buy")
    amanda_careers: tuple[str, ...] = ("pharma", "wet_lab", "animal_science", "data_science")
    company_modes: tuple[str, ...] = ("direct", "business")
    company_outcomes: tuple[str, ...] = ("dormant", "saas", "licensing", "hybrid", "vc", "failure")
    macros: tuple[str, ...] = ("baseline", "recession", "stagflation", "ai_modest", "ai_substantial", "ai_extreme")
    exit_values: tuple[float, ...] = (0, 10e6, 25e6, 50e6, 100e6, 250e6, 500e6, 1e9)
    exit_types: tuple[str, ...] = ("equity", "asset")
    maximum_children: int = 3
    minimum_birth_spacing_months: int = 12
    funding_portfolios: tuple[tuple[str, ...], ...] = ((), ("darpa",), ("darpa", "sbir_phase1"), ("darpa", "foundation"), ("darpa", "pharma_agritech"), ("darpa", "ud_contract"), ("darpa", "sbir_phase1", "foundation", "pharma_agritech", "ud_contract"))
    grant_cases: tuple[str, ...] = ("awarded", "delayed", "reduced", "failed")
    benefits: tuple[str, ...] = ("retained", "lost")


@dataclass
class SalaryRange:
    minimum: float
    mode: float
    maximum: float


@dataclass
class SalarySupportPolicy:
    """Dedicated salary-only counterfactual, separate from career offer priors."""
    salaries: tuple[float, ...] = (*range(100_000, 650_000, 50_000), 800_000, 1_000_000)
    salary_start_year: int = 2027
    marriage_year: int = 2028
    retirement_age: int = 60
    reserve_months: float = 6
    company_support: str = "none"


@dataclass
class RunConfig:
    salary_support: SalarySupportPolicy = field(default_factory=SalarySupportPolicy)
    career_salary_ranges: dict[str, SalaryRange] = field(default_factory=lambda: {
        "bio_ai": SalaryRange(200_000, 250_000, 600_000),
        "pharma_agritech": SalaryRange(100_000, 300_000, 500_000),
    })
    decision_risk_budget: float = .05
    decision_minimum_paths: int = 32
    location_overrides: dict[str, dict] = field(default_factory=dict)
    career_salary_overrides: dict[str, float] = field(default_factory=dict)
    faculty_summer_months: float = 2
    faculty_summer_support_fraction: float = .5
    school_age_childcare_fraction: float = .4
    household: Household = field(default_factory=Household)
    housing: Housing = field(default_factory=Housing)
    business: Business = field(default_factory=Business)
    economics: Economics = field(default_factory=Economics)
    taxes: TaxPolicy = field(default_factory=TaxPolicy)
    retirement: RetirementPolicy = field(default_factory=RetirementPolicy)
    grid: Grid = field(default_factory=Grid)
    start_year: int = 2027
    end_year: int = 2036
    paths: int = 256
    seed: int = 20260910
    batch_size: int = 16
    threads: int = 8
    forecast_paths: int = 4096
    # These are subjective macro weights, not frequencies inferred from history.
    macro_weights: tuple[float, ...] = (.45, .15, .10, .15, .10, .05)
    forecast_career_weights: dict[str, float] = field(default_factory=lambda: {"ud_blend": .6, "ud": .15, "synthyra": .1, "protein_design": .1, "faculty_ud": .05})
    ledger_paths: int = 3
    generate_reports: bool = True

    def validate(self) -> None:
        support = self.salary_support
        if not support.salaries or any(v <= 0 for v in support.salaries) or list(support.salaries) != sorted(set(support.salaries)):
            raise ValueError("Salary support grid must contain increasing positive salaries")
        if support.reserve_months < 0 or support.retirement_age <= 0:
            raise ValueError("Invalid salary-support reserve or retirement target")
        if not 0 < self.decision_risk_budget < 1 or self.decision_minimum_paths < 2:
            raise ValueError("Invalid decision risk budget or minimum path count")
        for name, salary in self.career_salary_ranges.items():
            if not 0 < salary.minimum <= salary.mode <= salary.maximum or salary.minimum == salary.maximum:
                raise ValueError(f"Invalid salary distribution: {name}")
        if self.start_year != 2027 or self.end_year < self.start_year:
            raise ValueError("The September 2026 bridge requires start_year=2027 and end_year >= 2027")
        if self.paths < 2 or self.batch_size < 1 or self.threads < 1:
            raise ValueError("Use at least 2 paths, one batch item, and one CPU thread")
        r = self.retirement
        if not (0 < r.success_target < 1 and 0 < r.withdrawal_rate <= 1 and 0 <= r.withdrawal_tax_rate < 1):
            raise ValueError("Invalid retirement probability, withdrawal rate or tax assumption")
        if r.maximum_retirement_age >= r.life_expectancy or r.maximum_retirement_age <= self.household.logan.age or r.retirement_paths < 2:
            raise ValueError("Retirement age must precede life expectancy; use at least two retirement paths")
        if r.real_return <= -1 or min(r.annual_volatility, r.annual_spending_floor, r.annual_healthcare) < 0 or (r.annual_spending_override is not None and r.annual_spending_override <= 0):
            raise ValueError("Invalid retirement return or spending assumptions")
        if not 0 <= r.college_start_age < r.child_support_end_age or r.college_annual_cost < 0 or not 0 <= r.college_parent_share <= 1 or r.future_inflation <= -1:
            raise ValueError("Invalid child-support, college or future-inflation assumption")
        if self.housing.mortgage_years < 1:
            raise ValueError("Mortgage term must be positive")
        if not self.grid.years or any(y < self.start_year or y > self.end_year for y in self.grid.years):
            raise ValueError("Grid years must be nonempty and within the horizon")
        if not 1 <= self.grid.event_month <= 12 or not 0 <= self.grid.maximum_children <= 3:
            raise ValueError("Invalid event month or child count")
        for loan in self.household.loans:
            if min(loan.balance, loan.apr, loan.minimum) < 0 or loan.repayment_months < 1:
                raise ValueError(f"Invalid loan: {loan.name}")
        if self.business.tax_classification not in ("partnership", "c_corp"):
            raise ValueError("Supported tax classifications: partnership, c_corp")
        for grant in self.business.grants:
            if grant.months < 1 or grant.amount < 0 or not 0 <= grant.award_probability <= 1:
                raise ValueError(f"Invalid grant: {grant.name}")
            if grant.milestones and abs(sum(f for _, f in grant.milestones) - 1) > 1e-8:
                raise ValueError(f"Milestone fractions must sum to 1: {grant.name}")
        if not 0 <= self.business.exit_escrow_fraction + self.business.exit_earnout_fraction <= 1:
            raise ValueError("Exit holdbacks exceed proceeds")


def config_dict(config: RunConfig) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(config)))


def config_from_dict(raw: dict) -> RunConfig:
    raw = json.loads(json.dumps(raw))
    if "career_salary_ranges" in raw:
        raw["career_salary_ranges"] = {key: SalaryRange(**value) for key, value in raw["career_salary_ranges"].items()}
    household = raw.pop("household", {})
    for key in ("logan", "amanda"):
        if key in household:
            household[key] = Person(**household[key])
    if "loans" in household:
        household["loans"] = [Loan(**v) for v in household["loans"]]
    business = raw.pop("business", {})
    for key, cls in (("grants", Grant), ("rounds", FundingRound)):
        if key in business:
            business[key] = [cls(**v) for v in business[key]]
    config = RunConfig(salary_support=SalarySupportPolicy(**raw.pop("salary_support", {})),
                       household=Household(**household), business=Business(**business),
                       housing=Housing(**raw.pop("housing", {})), economics=Economics(**raw.pop("economics", {})),
                       taxes=TaxPolicy(**raw.pop("taxes", {})), retirement=RetirementPolicy(**raw.pop("retirement", {})),
                       grid=Grid(**raw.pop("grid", {})), **raw)
    config.validate()
    return config


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    if path.suffix == ".json":
        config = config_from_dict(json.loads(path.read_text(encoding="utf-8")))
        if config.economics.history_csv:
            history = Path(config.economics.history_csv)
            if not history.is_absolute():
                nearby = path.parent / history
                config.economics.history_csv = str((nearby if nearby.exists() else history).resolve())
        return config
    spec = importlib.util.spec_from_file_location("user_finance_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load configuration: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.CONFIG
    if not isinstance(config, RunConfig):
        raise TypeError("CONFIG must be a RunConfig")
    config.validate()
    return config
