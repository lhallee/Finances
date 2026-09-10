"""Annual tax calculations, compiled for repeated path evaluations.

Federal rules use 2026 anchors. State schedules and future indexing are planning
approximations, not implementations of complete individual state return forms.
"""

import numpy as np

from numba import njit


@njit(cache=True)
def progressive(income: float, bounds: np.ndarray, rates: np.ndarray) -> float:
    # bounds, rates: (k,), each bound is the start of its marginal bracket.
    tax = 0.0
    for i in range(len(bounds)):
        upper = bounds[i + 1] if i + 1 < len(bounds) else max(income, bounds[i])
        tax += max(0.0, min(income, upper) - bounds[i]) * rates[i]
    return tax


@njit(cache=True)
def payroll_tax(wages: float, self_employment: float, base: float) -> float:
    net_se = max(0.0, self_employment) * .9235
    if net_se < 400:
        net_se = 0.0
    return .062 * min(base, max(0.0, wages)) + .0145 * max(0.0, wages) + .124 * min(max(0.0, base - wages), net_se) + .029 * net_se


@njit(cache=True)
def state_income_tax(income: float, state: str, joint: bool, scale: float) -> float:
    multiplier = 2.0 if joint else 1.0
    if state == "PA":
        return max(0.0, income) * .0307
    if state == "DE":
        taxable = max(0.0, income - 3250 * multiplier)
        bounds = np.array([0., 2000, 5000, 10_000, 20_000, 25_000, 60_000])  # (7,)
        rates = np.array([0., .022, .039, .048, .052, .0555, .066])  # (7,)
        return max(0.0, progressive(taxable, bounds, rates) - 110 * multiplier)
    if state == "ME":
        bounds = np.array([0., 26_800, 63_450]) * multiplier * scale  # (3,)
        rates = np.array([.058, .0675, .0715])  # (3,)
        return progressive(max(0.0, income - 16_100 * multiplier * scale), bounds, rates)
    if state == "NY":
        bounds = np.array([0., 8500, 11_700, 13_900, 80_650, 215_400, 1_077_550, 5e6, 25e6]) * multiplier * scale  # (9,)
        rates = np.array([.04, .045, .0525, .055, .06, .0685, .0965, .103, .109])  # (9,)
        return progressive(max(0.0, income - 8000 * multiplier), bounds, rates)
    if state == "CA":
        bounds = np.array([0., 11_079, 26_264, 41_452, 57_542, 72_724, 371_479, 445_771, 742_953]) * multiplier * scale  # (9,)
        rates = np.array([.01, .02, .04, .06, .08, .093, .103, .113, .123])  # (9,)
        taxable = max(0.0, income - 5706 * multiplier * scale)
        return progressive(taxable, bounds, rates) + max(0.0, taxable - 1e6) * .01
    return max(0.0, income - 4400 * multiplier) * .05 + max(0.0, income - 1_083_150 * scale) * .04


@njit(cache=True)
def federal_income_tax(ordinary: float, gains: float, qualified_dividends: float,
                       joint: bool, children: int, earned1: float, earned2: float,
                       childcare: float, scale: float, itemized: float = 0.0,
                       standard_single: float = 16_100., child_credit: float = 2200.) -> float:
    multiplier = 2.0 if joint else 1.0
    deduction = max(standard_single * multiplier * scale, itemized)
    taxable = max(0.0, ordinary - deduction)
    bounds = np.array([0., 12_400, 50_400, 105_700, 201_775, 256_225, 640_600]) * scale  # (7,)
    if joint:
        bounds = np.array([0., 24_800, 100_800, 211_400, 403_550, 512_450, 768_700]) * scale  # (7,)
    rates = np.array([.10, .12, .22, .24, .32, .35, .37])  # (7,)
    tax = progressive(taxable, bounds, rates)
    preferential = max(0.0, gains + qualified_dividends - max(0.0, deduction - ordinary))
    zero_end = (98_900 if joint else 49_450) * scale
    fifteen_end = (613_700 if joint else 545_500) * scale
    zero_part = min(preferential, max(0.0, zero_end - taxable))
    fifteen_part = min(preferential - zero_part, max(0.0, fifteen_end - max(zero_end, taxable)))
    tax += .15 * fifteen_part + .20 * max(0.0, preferential - zero_part - fifteen_part)
    agi = ordinary + gains + qualified_dividends
    phaseout = np.ceil(max(0.0, agi - (400_000 if joint else 200_000)) / 1000) * 50
    credit = max(0.0, children * (50 * np.floor(child_credit * scale / 50)) - phaseout)
    refundable = min(children * 1700, max(0.0, earned1 + earned2 - 2500) * .15, max(0.0, credit - tax))
    tax = max(0.0, tax - credit) - refundable
    # Earned-income requirement makes this zero when one spouse stops all year.
    allowed_care = min(childcare, 6000 if children > 1 else 3000, max(0.0, earned1))
    if joint:
        allowed_care = min(allowed_care, max(0.0, earned2))
    care_rate = max(.35, .50 - .01 * np.ceil(max(0.0, agi - 15_000) / 2000))
    care_rate = max(.20, care_rate - .01 * np.ceil(max(0.0, agi - (150_000 if joint else 75_000)) / (4000 if joint else 2000)))
    return tax - min(max(0.0, tax), allowed_care * care_rate)


@njit(cache=True)
def annual_tax(wage1: float, wage2: float, se1: float, other: float, gains: float,
               dividends: float, retirement1: float, retirement2: float,
               pretax_health: float, joint: bool, children: int, childcare: float,
               state: str, local_rate: float, de_source: float, scale: float,
               mortgage_interest: float = 0.0, property_tax: float = 0.0,
               qsbs_excluded: float = 0.0, state_qsbs: bool = False,
               ss_base: float = 184_500., standard_single: float = 16_100.,
               child_credit: float = 2200., tax_year: int = 2026) -> tuple:
    base = ss_base * scale
    wage1 = max(0.0, wage1)
    wage2 = max(0.0, wage2)
    # Health is assigned to Logan for payroll exclusion, capped by his wages.
    health1 = min(wage1, pretax_health)
    health2 = min(wage2, max(0.0, pretax_health - health1))
    fica = payroll_tax(wage1 - health1, se1, base) + payroll_tax(wage2 - health2, 0.0, base)
    earnings = wage1 + wage2 - pretax_health + max(0.0, se1) * .9235
    if joint:
        fica += .009 * max(0.0, earnings - 250_000)
    else:
        fica += .009 * max(0.0, wage1 - health1 + max(0.0, se1) * .9235 - 200_000)
        fica += .009 * max(0.0, wage2 - health2 - 200_000)
    se_deduction = .5 * (payroll_tax(wage1 - health1, se1, base) - payroll_tax(wage1 - health1, 0.0, base))
    ordinary1 = wage1 + se1 - retirement1 - health1 - se_deduction + other
    ordinary2 = wage2 - retirement2 - health2
    taxable_gain = max(0.0, gains - qsbs_excluded)
    state_base = max(0.0, ordinary1 + ordinary2 + gains + dividends - (qsbs_excluded if state_qsbs else 0.0))
    if state == "PA":
        state_base += retirement1 + retirement2
    if joint:
        state_tax = state_income_tax(state_base, state, True, scale)
    else:
        amanda_share = max(0.0, ordinary2) / max(1.0, state_base)
        state_tax = state_income_tax(state_base * (1 - amanda_share), state, False, scale) + state_income_tax(state_base * amanda_share, state, False, scale)
    if state != "DE" and de_source > 0:
        de_tax = state_income_tax(max(0.0, ordinary1), "DE", False, scale) * min(1.0, de_source / max(1.0, wage1 + se1))
        resident_credit = min(state_tax * min(1.0, de_source / max(1.0, state_base)), de_tax)
        state_tax += de_tax - resident_credit
    local = (wage1 + wage2 + max(0.0, se1)) * local_rate
    state_tax += local
    if state == "CA":
        fica += .013 * (wage1 + wage2)  # Provisional CA SDI rate.
    salt_index = 1.01 ** max(0, tax_year - 2026)
    salt_limit = 10_000. if tax_year >= 2030 else max(10_000., 40_400 * salt_index - .30 * max(0., state_base - 505_000 * salt_index))
    itemized = mortgage_interest + min(salt_limit, max(0.0, state_tax) + property_tax)
    if joint:
        federal = federal_income_tax(ordinary1 + ordinary2, taxable_gain, dividends, True,
                                     children, wage1 + se1, wage2, childcare, scale, itemized, standard_single, child_credit)
        niit = .038 * min(max(0.0, other + taxable_gain + dividends), max(0.0, ordinary1 + ordinary2 + taxable_gain + dividends - 250_000))
    else:
        # Dependents assigned to Logan; separately configurable filing strategies are out of scope.
        federal = federal_income_tax(ordinary1, taxable_gain, dividends, False, children,
                                     wage1 + se1, 0.0, childcare, scale, itemized, standard_single, child_credit)
        federal += federal_income_tax(ordinary2, 0.0, 0.0, False, 0, wage2, 0.0, 0.0, scale, 0., standard_single, child_credit)
        niit = .038 * min(max(0.0, other + taxable_gain + dividends), max(0.0, ordinary1 + taxable_gain + dividends - 200_000))
    return federal + niit, state_tax, fica
