"""Dated reference anchors and explicit provisional geographic estimates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    state: str
    rent: float
    home_price: float
    property_tax: float
    cost_factor: float
    childcare: float
    local_tax: float = 0
    amanda_factor: float = 1


# Housing/childcare are editable planning estimates, not retrieved local quotes.
# Legacy locations remain readable for immutable older outputs. New grids use
# only the eight areas in Grid.locations.
LOCATIONS = {
    "parents": Location("PA", 0, 0, .015, 1, 1600),
    "squirrel_hill": Location("PA", 2300, 500_000, .022, 1.02, 1500, .03, .98),
    "bethel": Location("ME", 2000, 420_000, .015, 1.08, 1300, 0, .85),
    "norway": Location("ME", 1700, 330_000, .017, 1.03, 1200, 0, .85),
    "farmington": Location("ME", 1600, 310_000, .02, 1.02, 1200, 0, .85),
    "newark": Location("DE", 2200, 400_000, .01, 1, 1600),
    "philadelphia": Location("PA", 2400, 480_000, .018, 1.06, 1700, 0, 1.05),
    "nyc": Location("NY", 4500, 1_000_000, .014, 1.45, 2600, .038, 1.25),
    "bay_area": Location("CA", 4300, 1_400_000, .012, 1.4, 2500, 0, 1.35),
    "san_diego": Location("CA", 3400, 950_000, .012, 1.25, 2100, 0, 1.20),
    "boston": Location("MA", 3600, 850_000, .011, 1.30, 2400, 0, 1.2),
    "amherst": Location("MA", 2300, 480_000, .016, 1.08, 1800, 0, .95),
}

CAREER_SALARY = {"ud": 70_000, "ud_blend": 70_000, "ud_nonrenewal": 70_000,
                 "synthyra": 0, "biohub_engineer": 270_000, "biohub_scientist": 290_000,
                 "bio_ai_engineer": 270_000, "bio_ai_research": 290_000, "protein_design": 260_000,
                 "profluent": 260_000, "merck": 145_000, "faculty_ud": 110_000,
                 # Editable compensation cases, not employer offers or a market survey.
                 "pharma_agritech_pi": 190_000, "pharma_agritech_director": 250_000,
                 "faculty_ca": 145_000, "faculty_ma": 125_000,
                 "industry_blend": 180_000, "faculty_blend": 110_000}
AMANDA_FACTORS = {"pharma": 1, "wet_lab": .98, "animal_science": .9, "data_science": 1.3}
MACROS = {"baseline": (0, 0, 1), "recession": (-.12, -.005, 3),
          "stagflation": (-.04, .035, 1.8), "ai_modest": (.005, 0, 1.05),
          "ai_substantial": (.015, -.005, 1.5), "ai_extreme": (.02, -.01, 3)}
STATES = {"PA": 0, "DE": 1, "ME": 2, "NY": 3, "CA": 4, "MA": 5}

SOURCES = [
    {"id": "college_budget", "url": "https://nces.ed.gov/ipeds/search/viewtable?returnUrl=/search&tableId=36306", "as_of": "2026-09-10", "status": "published_components_with_editable_projection", "use": "NCES 2023-24 public four-year institutional averages in 2023 dollars: in-state tuition 8961, books 1260, on-campus room/board 12069, other expenses 3961. Components are not enrollment-weighted and need not equal reported total attendance cost. Rounded planning proxy 28500 in 2026 dollars, approximately three years at assumed 2.5% inflation. User covers half for ages 18-21; college allowance replaces regular child support."},
    {"id": "salary_landscapes", "url": "", "as_of": "2026-09-10", "status": "user_specified_range_modeler_selected_mode", "use": "Annual 2026-dollar salary: bio-AI $200k-$600k triangular mode $250k; pharma/agritech $100k-$500k triangular mode $300k. These distributions are user planning assumptions, not measured wage distributions. Independent location choices may require unconfirmed employer permission."},
    {"id": "industry_leadership", "url": "MODEL.md", "as_of": "2026-09-10", "status": "editable_planning_assumption", "use": "Pharma/agritech principal-scientist/PI-adjacent and research-director cases use $190k/$250k base pay; employer-specific offers, bonuses, equity, and regional calibration remain unresolved"},
    {"id": "retirement_policy", "url": "MODEL.md#retirement-readiness", "as_of": "2026-09-10", "status": "editable_planning_assumption", "use": "Longevity 100, 95% target, 4% withdrawal cap, 3% real return, 15% volatility, flat 20% withdrawal tax reserve; none are guarantees or calibrated retirement advice"},
    {"id": "care_credit_2026", "url": "https://usc-cdn.house.gov/view.xhtml?edition=prelim&req=granuleid%3AUSC-prelim-title26-section21", "as_of": "2026-09-10", "status": "published_rule", "use": "2026 dependent-care credit with 50%, 35%, 20% thresholds and earned-income limits"},
    {"id": "salt", "url": "https://www.irs.gov/pub/irs-access/p4491_accessible.pdf", "as_of": "2026-09-10", "status": "published_rule", "use": "Temporary expanded SALT deduction through 2029, then $10k; 1% statutory indexing before expiration"},
    {"id": "user", "url": "", "as_of": "2026-09-10", "status": "user_reported", "use": "Household starting finances, DARPA 95% subjective prior, company cash/costs, ownership"},
    {"id": "prior_workbook", "url": "PROJECT_CONTEXT.md", "as_of": "2026-09-10", "status": "consolidated_user_context", "use": "Imported workbook loans, cash, spending and saving policy; original contextual folders removed at user request"},
    {"id": "irs_2026", "url": "https://www.irs.gov/newsroom/irs-releases-tax-inflation-adjustments-for-tax-year-2026-including-amendments-from-the-one-big-beautiful-bill", "as_of": "2026-09-10", "status": "published_anchor", "use": "2026 federal brackets and deductions; later indexed values are projections"},
    {"id": "ssa", "url": "https://www.ssa.gov/OACT/COLA/cbb.html", "as_of": "2026-09-10", "status": "published_anchor", "use": "2026 Social Security wage base"},
    {"id": "ud_rates", "url": "https://www.udel.edu/faculty-staff/human-resources/total-rewards/rates/", "as_of": "2026-09-10", "status": "prior_handoff_verified", "use": "Employee-only premiums; family premiums provisional; 45% FTE eligibility unknown"},
    {"id": "ud_retirement", "url": "https://www.udel.edu/faculty-staff/human-resources/total-rewards/retirement-information/403b-retirement-income/", "as_of": "2026-09-10", "status": "prior_handoff_verified", "use": "5% employee / 11% UD contribution conditional on eligibility"},
    {"id": "biohub", "url": "https://job-boards.greenhouse.io/biohub/jobs/7747517", "as_of": "2026-09-10", "status": "published_range", "use": "Advertised $214k-$375k; selected salary and offer odds are assumptions"},
    {"id": "profluent", "url": "https://job-boards.greenhouse.io/profluent/jobs/5248524008", "as_of": "2026-09-10", "status": "published_range", "use": "Advertised $200k-$330k; selected salary is an assumption"},
    {"id": "bls", "url": "https://www.bls.gov/oes/tables.htm", "as_of": "2026-09-10", "status": "calibration_source_not_ingested", "use": "Occupation/location wage calibration; Amanda and faculty factors currently provisional"},
    {"id": "housing", "url": "https://www.zillow.com/research/data/", "as_of": "2026-09-10", "status": "calibration_source_not_ingested", "use": "Housing estimates are provisional; replace with geography-specific observations"},
    {"id": "childcare", "url": "https://www.dol.gov/agencies/wb/topics/featured-childcare", "as_of": "2026-09-10", "status": "calibration_source_not_ingested", "use": "County/provider/age childcare; configured values currently provisional"},
    {"id": "ai", "url": "https://www.anthropic.com/institute/econ-scenarios", "as_of": "2026-09-10", "status": "conditional_narrative", "use": "AI scenario categories; financial offsets and probabilities are subjective"},
    {"id": "partnership", "url": "https://www.irs.gov/faqs/small-business-self-employed-other-business/entities/entities-1", "as_of": "2026-09-10", "status": "published_rule", "use": "Partner services are self-employment rather than W-2 wages"},
    {"id": "qsbs", "url": "https://uscode.house.gov/view.xhtml?req=%28title%3A26+section%3A1202+edition%3Aprelim%29", "as_of": "2026-09-10", "status": "conditional_rule", "use": "QSBS requires original-issue eligibility, holding period, basis and state review; disabled by default"},
    {"id": "pa", "url": "https://www.pa.gov/agencies/revenue/resources/tax-types-and-information/personal-income-tax", "as_of": "2026-09-10", "status": "planning_rule", "use": "3.07%, retirement deferrals taxable; local rates need address verification"},
    {"id": "de", "url": "https://revenue.delaware.gov/employers-guide-withholding-regulations-employers-duties/", "as_of": "2026-09-10", "status": "planning_rule", "use": "Delaware brackets and source-share approximation"},
    {"id": "me", "url": "https://www.maine.gov/revenue/taxes/income-estate-tax/individual-income-tax", "as_of": "2026-09-10", "status": "approximation", "use": "Projected state schedule; credits require refinement"},
    {"id": "ny", "url": "https://www.tax.ny.gov/pit/", "as_of": "2026-09-10", "status": "approximation", "use": "State brackets; NYC flat effective local estimate; recapture not implemented"},
    {"id": "ca", "url": "https://www.ftb.ca.gov/file/personal/tax-calculator-tables-rates.asp", "as_of": "2026-09-10", "status": "approximation", "use": "Projected brackets and millionaire surtax; SDI modeled separately"},
    {"id": "ma", "url": "https://www.mass.gov/personal-income-tax", "as_of": "2026-09-10", "status": "approximation", "use": "5% ordinary and modeled long-term gains, 4% surtax; short-term gain differentiation not implemented"},
]

LIMITATIONS = [
    "This is a planning model, not a complete tax-return engine. MODEL.md lists unimplemented tax and calibration requirements.",
    "AMT, QBI, capital-loss carryforwards, dependent-care FSA and exact pre-marriage investment tax ownership are not implemented.",
    "Amanda balances, benefits and spending are provisional estimates.",
    "Housing, childcare, faculty pay and regional salary factors are provisional, not measured confidence intervals.",
    "2027 onward tax thresholds are projected from reference rules; state credits, NY recapture and some itemized limitations are approximate.",
    "QSBS eligibility is disabled by default. Tax basis, LLC hot assets and state conformity require confirmation.",
    "Partial-year state tax uses income allocation, not complete state return forms. Delaware remote source share is explicit.",
    "No disability/death/divorce model, estate planning, wash sales, loss harvesting, or automatic credit facility.",
    "Economic mode is saved with each run. Historical mode uses aligned block resampling; parametric mode uses correlated stress assumptions.",
    "Company/career probabilities and operating assumptions are subjective; predicted valuations are not observed company values.",
    "Grant budgets and eligibility are modeling inputs, not a determination of program compliance.",
    "September balances are carried to October 1 without an unobserved September partial-month transaction estimate.",
]
