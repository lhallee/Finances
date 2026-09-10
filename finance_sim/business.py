"""Company cash, restricted project reimbursements, financing, and founder proceeds."""

import numpy as np

from numba import njit, prange
from collections import namedtuple

from .configuration import RunConfig
from .parameters import BusinessParameters
from .economics import monthly_dates
from .equity import dilute, exit_waterfall, qsbs_exclusion
from .scenarios import Scenario
from .schedule import SCHEDULE


BUSINESS_COLUMNS = ("founder_pay", "company_cash", "revenue", "operating_cost", "grant_receipts",
                    "financing", "founder_share", "equity_proceeds", "ordinary_proceeds",
                    "qsbs_exclusion", "company_arrears", "distribution", "c_corp",
                    "company_value", "advisor_payout", "exit_gross", "grant_claims", "staff_fte",
                    "restricted_cash", "company_receivables", "company_tax", "exit_basis_recovered", "darpa_awarded")

BUSINESS = namedtuple("BusinessColumns", BUSINESS_COLUMNS)(*range(len(BUSINESS_COLUMNS)))


def pack_business(config: RunConfig, scenario: Scenario) -> tuple:
    b = config.business
    dates = monthly_dates(config)
    start = dates[0]
    grant_rows = []
    for grant in b.grants:
        if grant.name not in scenario.grants:
            continue
        date = np.datetime64(grant.start, "M")
        offset = int((date - np.datetime64(start.strftime("%Y-%m"), "M")).astype(int))
        grant_rows.append([grant.amount, offset, grant.months, grant.award_probability,
                           grant.receipt_delay_months, grant.effort_fraction, grant.cost_share,
                           grant.overhead_rate, grant.direct_cost_monthly, grant.follow_on_probability,
                           grant.follow_on_annual_revenue, grant.follow_on_months,
                           grant.delay_probability, grant.delayed_months, grant.equity_fraction, float(grant.name == "darpa")])
    grants = np.asarray(grant_rows, dtype=float).reshape(-1, 16)  # (g, 15)
    milestones = np.zeros((len(grants), len(dates)), dtype=float)  # (g, t)
    selected = [g for g in b.grants if g.name in scenario.grants]
    for i, grant in enumerate(selected):
        for offset, fraction in grant.milestones:
            month = int(grants[i, 1]) + offset
            if 0 <= month < len(dates):
                milestones[i, month] += fraction
    rounds_list = b.rounds if scenario.company_outcome == "vc" else []
    if len(rounds_list) > 8:
        raise ValueError("At most eight financing rounds are supported by the conversion-election waterfall")
    rounds = np.asarray([[r.year, r.amount, r.pre_money, r.pool_topup, r.preference,
                          float(r.participating), {"priced": 0, "safe": 1, "note": 2}[r.kind],
                          r.conversion_year, r.valuation_cap, r.discount, r.interest] for r in rounds_list], dtype=float).reshape(-1, 11)  # (r, 11)
    values = BusinessParameters(b.cash, b.monthly_revenue, b.monthly_overhead, b.starting_clients,
                       b.monthly_new_clients, b.monthly_churn, b.annual_revenue_per_client,
                       b.gross_margin, b.licensing_annual_probability, b.licensing_upfront,
                       b.licensing_annual_royalty, b.payroll_load, b.initial_staff_fte,
                       b.staff_annual_cost, b.max_staff_fte, b.founder_fulltime_salary,
                       b.vc_founder_salary, b.revenue_salary_fraction, b.salary_cap,
                       b.distributions_fraction, b.logan_equity * (1-b.braydon_issuance) * (1-b.diane_issuance),
                       b.stephen_participation, b.equity_basis, b.exit_transaction_cost,
                       b.exit_escrow_fraction, b.exit_escrow_months, b.exit_earnout_fraction,
                       b.exit_earnout_months, b.exit_earnout_probability, b.exit_debt,
                       float(b.qsbs_eligible), b.qsbs_conversion_value, b.llc_hot_asset_fraction,
                       float(b.tax_classification == "c_corp"), b.asset_basis,
                       int((np.datetime64(b.stephen_grant_date, "M") + np.timedelta64(12 * b.stephen_expiration_years + 1, "M") - np.datetime64(start.strftime("%Y-%m"), "M")).astype(int)), b.fulltime_founder_effort, b.annual_raise)
    values = BusinessParameters._make(map(float, values))
    return grants, rounds, values, milestones


@njit(cache=True, parallel=True)
def company_kernel(schedule: np.ndarray, draws: np.ndarray, grants: np.ndarray,
                   rounds: np.ndarray, values: BusinessParameters, milestones: np.ndarray,
                   mode: str, outcome: str, grant_case: str, exit_value: float,
                   asset_sale: bool, forecast: bool) -> np.ndarray:
    # schedule (t,26), draws (p,t,8), grants (g,15), rounds (r,11), values (37,), milestones (g,t).
    p, t = draws.shape[:2]
    output = np.zeros((p, t, len(BUSINESS_COLUMNS)))  # (p, t, business metric)
    for path in prange(p):
        cash, clients, share = values.cash, values.starting_clients, values.founder_share
        arrears, royalties, firm_value = 0.0, 0.0, 0.0
        retained_salary = 0.
        inflation, pay_months, salary_index = 1., 0, 1.
        corporate = values.corporate > .5
        converted_month, exited = -1, False
        investor_shares = np.zeros(len(rounds))  # (r,)
        invested = np.zeros(len(rounds))  # (r,)
        multiples = np.ones(len(rounds))  # (r,)
        participants = np.zeros(len(rounds))  # (r,)
        bridge_notes = np.zeros(len(rounds))  # (r,)
        escrow = np.zeros((t + 121, 4))  # (t+61, equity cash, ordinary cash, excluded gain)
        future_receipts = np.zeros(t + 121)  # (t+121,)
        remaining = grants[:, 0].copy()  # (g,)
        awarded = np.ones(len(grants))  # (g,)
        delays = np.zeros(len(grants), dtype=np.int64)  # (g,)
        restricted = 0.0
        advisor_scale = 1.0
        for g in range(len(grants)):
            u = draws[path, g % t, 7]
            if grant_case == "failed" or (forecast and u > grants[g, 3]):
                awarded[g] = 0
            if grant_case == "reduced":
                remaining[g] *= .5
            delays[g] = int(grants[g, 4]) + (int(grants[g, 13]) if grant_case == "delayed" or (forecast and draws[path, (g + 11) % t, 7] < grants[g, 12]) else 0)
        darpa_awarded = 1.
        for g in range(len(grants)):
            if grants[g, 15] > .5:
                darpa_awarded = awarded[g]
        for m in range(t):
            year, month = int(schedule[m, SCHEDULE.year]), int(schedule[m, SCHEDULE.month])
            inflation *= max(.9, 1 + draws[path, m, 1])
            financing, grant_cash, founder_pay, company_tax = 0., future_receipts[m], 0., 0.
            equity_cash, ordinary_cash, excluded = escrow[m, 0], escrow[m, 1], escrow[m, 2]
            gross_exit, advisor_pay = 0., 0.
            recovered_basis = escrow[m, 3]
            for r in range(len(rounds)):
                if year == int(rounds[r, 0]) and month == 1 and not exited:
                    financing += rounds[r, 1]
                    if not corporate:
                        corporate, converted_month = True, m
                    if rounds[r, 6] == 0:
                        old = share
                        share, investor_shares, fraction = dilute(share, investor_shares, rounds[r, 1], rounds[r, 2], rounds[r, 3])
                        investor_shares[r] += fraction
                        invested[r], multiples[r], participants[r] = rounds[r, 1], rounds[r, 4], rounds[r, 5]
                        advisor_scale *= share / max(old, 1e-12)
                    else:
                        bridge_notes[r] = rounds[r, 1]
                if bridge_notes[r] > 0 and year == int(rounds[r, 7]) and month == 1 and not exited:
                    principal = bridge_notes[r] * ((1 + rounds[r, 10]) ** max(0, year - rounds[r, 0]) if rounds[r, 6] == 2 else 1)
                    effective_pre = min(rounds[r, 8], rounds[r, 2] * (1 - rounds[r, 9]))
                    old = share
                    share, investor_shares, fraction = dilute(share, investor_shares, principal, effective_pre, 0.)
                    investor_shares[r] += fraction
                    invested[r], multiples[r], participants[r] = principal, rounds[r, 4], rounds[r, 5]
                    advisor_scale *= share / max(old, 1e-12)
                    bridge_notes[r] = 0
            cash += financing
            fte = schedule[m, SCHEDULE.founder_fte]
            if schedule[m, SCHEDULE.darpa_blend] > .5 and darpa_awarded < .5:
                fte = 0.
            active = not exited and not (outcome == "failure" and year >= 2029)
            revenue = values.monthly_revenue
            if active and year >= 2027 and outcome not in ("dormant", "failure"):
                demand = max(.05, 1 + draws[path, m, 0] * 2)
                clients = max(0., clients * (1 - values.monthly_churn) + values.monthly_new_clients * demand)
                if outcome in ("saas", "hybrid", "vc"):
                    revenue += clients * values.annual_revenue_per_client / 12 if mode == "business" else (25_000 if outcome == "vc" else 5000) * (1.10 ** (year - 2027))
                if outcome in ("licensing", "hybrid") and draws[path, m, 7] < values.licensing_probability / 12:
                    revenue += values.licensing_upfront
                    royalties += values.licensing_royalty / 12
                revenue += royalties
            if not active:
                revenue = 0
            direct, effort = 0., 0.
            for g in range(len(grants)):
                start, duration = int(grants[g, 1]), int(grants[g, 2])
                if awarded[g] and active and start <= m < start + duration:
                    effort += grants[g, 5]
                    direct += grants[g, 8] * inflation
                    if m == start and grants[g, 14] > 0:
                        retained = 1 - grants[g, 14]
                        share *= retained
                        investor_shares *= retained  # (r,)
                        advisor_scale *= retained
                if awarded[g] and active and start + duration <= m < start + duration + int(grants[g, 11]):
                    if draws[path, (g + 37) % t, 7] < grants[g, 9]:
                        revenue += grants[g, 10] / 12
            staff = min(values.max_staff, max(values.initial_staff, np.ceil(max(0., effort - min(fte, values.founder_effort)))))
            capacity_scale = min(1., (staff + min(fte, values.founder_effort)) / max(effort, 1e-12))
            direct *= capacity_scale
            if fte > 0:
                if pay_months > 0 and pay_months % 12 == 0:
                    salary_index *= 1 + values.annual_raise
                    retained_salary *= 1 + values.annual_raise
                pay_months += 1
            annual_salary = (values.vc_salary if corporate else values.founder_salary) * salary_index
            annual_salary = min(values.salary_cap * salary_index, max(annual_salary, revenue * 12 * values.revenue_salary_fraction))
            desired_pay = annual_salary / 12 * fte if active else 0
            overhead = values.monthly_overhead * inflation if active else 0
            # Initial unpaid founder/cofounder capacity is separate from incremental hired staff.
            hires = max(0., staff - values.initial_staff) * values.staff_cost * inflation / 12
            planned_cost = overhead + direct + hires + desired_pay * (1 + values.payroll_load) + revenue * (1 - values.gross_margin)
            claims = np.zeros(len(grants))  # (g,)
            immediate_claims = 0.
            advanced = 0.
            for g in range(len(grants)):
                start, duration = int(grants[g, 1]), int(grants[g, 2])
                if awarded[g] and active and start <= m < start + duration:
                    allocated_pay = desired_pay * min(1., grants[g, 5] / max(effort, 1e-12))
                    eligible = (grants[g, 8] * inflation * capacity_scale + allocated_pay * (1 + values.payroll_load)) * (1 + grants[g, 7]) * (1 - grants[g, 6])
                    claims[g] = min(remaining[g], eligible, grants[g, 0] / duration)
                    if np.sum(milestones[g]) > 0:
                        advanced += grants[g, 0] * milestones[g, m]
                    elif delays[g] == 0:
                        immediate_claims += claims[g]
            # Every claim must be backed by actual total spending, including overhead.
            claim_scale = min(1., planned_cost / max(1e-12, np.sum(claims)))
            claims *= claim_scale
            immediate_claims *= claim_scale
            # Restricted advances finance only eligible costs. Proportional spending
            # solves same-month reimbursement without claims for unfunded expenses.
            restricted += advanced
            restricted_use = min(restricted, np.sum(claims))
            available = max(0., cash + revenue + grant_cash + restricted_use)
            ratio = min(1., available / max(1e-12, planned_cost - immediate_claims))
            total_claims = 0.
            for g in range(len(grants)):
                claim = claims[g] * ratio
                remaining[g] -= claim
                total_claims += claim
                if np.sum(milestones[g]) == 0:
                    receipt_month = m + delays[g]
                    if receipt_month == m:
                        grant_cash += claim
                    elif receipt_month < len(future_receipts):
                        future_receipts[receipt_month] += claim
            restricted_use = min(restricted, total_claims)
            restricted -= restricted_use
            grant_cash += restricted_use
            cost = planned_cost * ratio
            founder_pay = desired_pay * ratio
            arrears += max(0., desired_pay - founder_pay)
            taxable_profit = max(0., revenue + total_claims - cost)
            if corporate:
                company_tax = taxable_profit * .26  # Federal 21% plus provisional state allowance.
            cash = max(0., cash + revenue + grant_cash - cost - company_tax)
            distribution = min(cash, taxable_profit * values.distribution_fraction) * share if not corporate else 0.
            if distribution > 0:
                cash -= distribution / max(share, 1e-12)
            firm_value = revenue * 12 * (12 if outcome == "vc" else 5) if active else 0.
            if schedule[m, SCHEDULE.exit] and not exited:
                debt = values.exit_debt + arrears + np.sum(bridge_notes)
                gross_exit, advisor_pay, investors, available_exit = exit_waterfall(exit_value, cash, debt, values.exit_fee, share, investor_shares, invested, multiples, participants,
                    values.advisor_fraction * advisor_scale if m < int(values.advisor_expiration_month) else 0.)
                if gross_exit < 0:
                    output[path, m, 15] = -1
                    continue
                corporate_asset_tax = .26 * max(0., exit_value - values.asset_basis) if asset_sale and corporate else 0.
                if corporate_asset_tax > 0:
                    gross_exit, advisor_pay, investors, available_exit = exit_waterfall(exit_value, cash, debt + corporate_asset_tax, values.exit_fee, share, investor_shares, invested, multiples, participants, values.advisor_fraction * advisor_scale if m < int(values.advisor_expiration_month) else 0.)
                    company_tax += corporate_asset_tax
                ordinary_part = max(0., gross_exit - values.equity_basis) * values.hot_asset_fraction if not corporate else 0.
                gain = max(0., gross_exit - values.equity_basis - ordinary_part)
                exclusion = qsbs_exclusion(gain, values.equity_basis, values.conversion_value, m - converted_month if converted_month >= 0 else 0, bool(values.qsbs_eligible) and corporate and not asset_sale)
                now = 1 - values.escrow_fraction - values.earnout_fraction
                equity_cash += (gross_exit - ordinary_part) * now
                ordinary_cash += ordinary_part * now
                excluded += exclusion * now
                exit_basis = min(values.equity_basis, max(0., gross_exit - ordinary_part))
                recovered_basis += exit_basis * now
                for is_earnout, delay, fraction in ((False, int(values.escrow_months), values.escrow_fraction), (True, int(values.earnout_months), values.earnout_fraction)):
                    if fraction <= 0:
                        continue
                    if is_earnout and draws[path, m, 7] > values.earnout_probability:
                        continue
                    due = m + delay
                    if due < len(escrow):
                        escrow[due, 0] += (gross_exit - ordinary_part) * fraction
                        escrow[due, 1] += ordinary_part * fraction
                        escrow[due, 2] += exclusion * fraction
                        escrow[due, 3] += exit_basis * fraction
                retained_salary = annual_salary
                future_receipts[:] = 0.
                restricted, arrears = 0., 0.
                exited = True
                cash, royalties, firm_value = 0., 0., 0.
            receivables = np.sum(future_receipts[m + 1:])
            if exited and not schedule[m, SCHEDULE.exit]:
                # Employment continues under the buyer. This payroll is outside
                # the sold company cash ledger and is treated as W-2 income.
                founder_pay = retained_salary / 12 * fte
                corporate = True
            output[path, m] = (founder_pay, cash, revenue, cost, grant_cash, financing, share,
                               equity_cash, ordinary_cash, excluded, arrears, distribution,
                               (1.0 if corporate else 0.0), firm_value * share, advisor_pay, gross_exit,
                               total_claims, staff, restricted, receivables, company_tax, recovered_basis, darpa_awarded)  # (21,)
    return output  # (p, t, 21)


def simulate_business(config: RunConfig, scenario: Scenario, schedule: np.ndarray,
                      draws: np.ndarray, forecast: bool = False) -> np.ndarray:
    # schedule (t,26), draws (p,t,8); result (p,t,21).
    grants, rounds, values, milestones = pack_business(config, scenario)
    result = company_kernel(schedule, draws, grants, rounds, values, milestones,
                            scenario.company_mode, scenario.company_outcome,
                            scenario.grant_case, scenario.exit_value, scenario.exit_type == "asset", forecast)  # (p,t,21)
    if np.any(result[:, :, 15] < 0):
        raise ValueError("No stable preferred-stock conversion election for this exit")
    return result  # (p,t,21)
