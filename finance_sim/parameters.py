"""Named immutable numerical parameters supported by the compiled kernels."""

from typing import NamedTuple


class BusinessParameters(NamedTuple):
    cash: float
    monthly_revenue: float
    monthly_overhead: float
    starting_clients: float
    monthly_new_clients: float
    monthly_churn: float
    annual_revenue_per_client: float
    gross_margin: float
    licensing_probability: float
    licensing_upfront: float
    licensing_royalty: float
    payroll_load: float
    initial_staff: float
    staff_cost: float
    max_staff: float
    founder_salary: float
    vc_salary: float
    revenue_salary_fraction: float
    salary_cap: float
    distribution_fraction: float
    founder_share: float
    advisor_fraction: float
    equity_basis: float
    exit_fee: float
    escrow_fraction: float
    escrow_months: float
    earnout_fraction: float
    earnout_months: float
    earnout_probability: float
    exit_debt: float
    qsbs_eligible: float
    conversion_value: float
    hot_asset_fraction: float
    corporate: float
    asset_basis: float
    advisor_expiration_month: float
    founder_effort: float
    annual_raise: float


class HouseholdParameters(NamedTuple):
    cash_target: float
    cash_cap: float
    debt_hurdle: float
    family_medical: float
    replacement_adult: float
    replacement_child: float
    domestic_partner: float
    child_other: float
    leave_pay: float
    medical_oop: float
    birth_cost: float
    wedding_cost: float
    job_loss_probability: float
    job_gap_months: float
    repair_probability: float
    repair_cost: float
    dividend_yield: float
    blog_subscribers: float
    blog_net: float
    prior_logan_wages: float
    prior_amanda_wages: float
    prior_withholding: float
    blog_growth: float
    debt_transition_band: float
    ud_job_loss_probability: float


class HousingParameters(NamedTuple):
    down_payment: float
    closing_cost: float
    sale_cost: float
    mortgage_rate: float
    mortgage_months: float
    reserve_months: float
    maintenance: float
    insurance: float
    pmi: float
    max_dti: float
    bedroom_premium: float
    moving_cost: float


