"""Copy to config.py. All fields are documented in finance_sim/configuration.py.

The full preset enumerates the entire configured grid and can be enormous.
Run --dry-run first. --preset demo is a separate, explicitly smaller grid.
Python configurations are trusted local code; do not load an unknown file.
"""

from finance_sim.configuration import RunConfig


CONFIG = RunConfig()
# Examples:
# CONFIG.paths = 1024
# CONFIG.grid.years = (2027, 2029, 2031)
# CONFIG.grid.careers = ("ud_blend", "protein_design")
# CONFIG.household.amanda.cash = 10_000  # PROVISIONAL, replace with actual balance
# CONFIG.household.domestic_partner_eligible = False
# CONFIG.business.qsbs_eligible = False  # Do not enable without confirming eligibility.
# CONFIG.retirement.annual_spending_override = 70_000  # Total real household retirement budget.
# CONFIG.retirement.success_target = .95
# CONFIG.retirement.life_expectancy = 100
# CONFIG.retirement.maximum_retirement_age = 80
# CONFIG.grid.practical_transition_years = (2027, 2029, 2032)

# Compensation distributions are user planning assumptions, in 2026 dollars.
# CONFIG.career_salary_ranges["bio_ai"].mode = 250_000  # Range: 200k to 600k.
# CONFIG.career_salary_ranges["pharma_agritech"].mode = 300_000  # Range: 100k to 500k.
# CONFIG.career_salary_overrides["protein_design"] = 400_000  # Exact selected rerun.
# CONFIG.decision_risk_budget = .05
# CONFIG.decision_minimum_paths = 32
# CONFIG.grid.independent_career_locations = True

# Dedicated zero-Synthyra salary support test:
# CONFIG.salary_support.retirement_age = 60
# CONFIG.salary_support.company_support = "none"  # Or "darpa_only": outside employment after the award.
# CONFIG.salary_support.reserve_months = 6
# CONFIG.salary_support.marriage_year = 2028
# CONFIG.salary_support.salary_start_year = 2027
# CONFIG.salary_support.salaries = (200_000, 300_000, 400_000, 500_000, 600_000)

# Lifecycle costs beyond the ten-year household simulation:
# CONFIG.retirement.child_support_end_age = 22
# CONFIG.retirement.college_start_age = 18
# CONFIG.retirement.college_annual_cost = 28_500  # Total tuition/living, 2026 dollars.
# CONFIG.retirement.college_parent_share = .5
# CONFIG.retirement.future_inflation = .025
# CONFIG.housing.mortgage_years = 30  # Also supports 20 or 25 years.
