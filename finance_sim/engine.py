"""Monthly household ledger with separate ownership and annual tax reconciliation."""

import numpy as np

from numba import njit, prange

from .business import BUSINESS, BUSINESS_COLUMNS, simulate_business
from .configuration import RunConfig
from .compensation import starting_salaries
from .parameters import HouseholdParameters, HousingParameters
from .economics import economic_paths
from .reference import CAREER_SALARY, MACROS, STATES
from .schedule import SCHEDULE, build_schedule
from .scenarios import Scenario
from .taxes import annual_tax


STATE_CODES = ("PA", "DE", "ME", "NY", "CA", "MA")


METRICS = ("cash", "brokerage", "brokerage_basis", "retirement", "debt", "home_value",
           "mortgage", "tax_payable", "unpaid_bills", "liquid_wealth", "financial_net_worth",
           "net_worth", "business_interest", "logan_income", "amanda_income", "company_pay",
           "living_cost", "childcare", "child_other", "health_premium", "housing_cost",
           "tax_expense", "federal_tax", "state_tax", "payroll_tax", "employee_retirement",
           "employer_retirement", "debt_payment", "debt_interest", "cash_interest", "realized_gains",
           "exit_cash", "shortfall", "purchased", "inflation_index", "reconciliation_error",
           "logan_cash", "amanda_cash", "logan_retirement", "amanda_retirement",
           "amanda_employee_retirement", "amanda_employer_retirement", "children", "amanda_stopped",
           "married", "earned_income", "investment_return", "home_equity", "tax_settlement",
           "mortgage_interest", "property_tax", "brokerage_contribution", "debt_extra",
           "distribution", "mortgage_payment", "medical_out_of_pocket")


@njit(cache=True)
def payment(principal: float, monthly_rate: float, months: int) -> float:
    if principal <= 0:
        return 0.
    if monthly_rate == 0:
        return principal / max(1, months)
    return principal * monthly_rate / (1 - (1 + monthly_rate) ** -max(1, months))


@njit(cache=True, parallel=True)
def household_kernel(schedule: np.ndarray, draws: np.ndarray, company: np.ndarray,
                     persons: np.ndarray, loans: np.ndarray, settings: HouseholdParameters,
                     home: HousingParameters, tax_settings: np.ndarray, job_multiplier: float,
                     salary_multipliers: np.ndarray) -> np.ndarray:
    # schedule (t,26), draws (p,t,8), company (p,t,21), persons (2,9),
    # loans (d,6), settings (23,), home (12,), tax_settings (7,).
    p, t = draws.shape[:2]
    output = np.zeros((p, t, len(METRICS)))  # (p,t,k)
    for path in prange(p):
        cash = persons[:, 0].copy()  # (2,)
        broker = persons[:, 1].copy()  # (2,)
        basis = persons[:, 2].copy()  # (2,)
        retirement = persons[:, 3].copy()  # (2,)
        balance = loans[:, 0].copy()  # (d,)
        accrued_deferred = np.zeros(len(loans))  # (d,) simple interest until capitalization
        minimums = loans[:, 2].copy()  # (d,)
        unpaid = np.zeros(2)  # (2,)
        tax_due, tax_current, inflation, wage_index = 0., 0., 1., 1.
        amanda_wage_index, pending_raise, amanda_pending_raise = 1., 1., 1.
        tax_arrears = 0.
        house_value, mortgage, mortgage_pay = 0., 0., 0.
        home_index = 1.
        employed_gap = np.zeros(2, dtype=np.int64)  # (2,)
        ytd = np.zeros(13)  # (13,) tax bases: wages1/2,se,other,gains,div,deferrals1/2,health,care,DE,interest,property.
        ytd[0], ytd[1] = settings.prior_logan_wages, settings.prior_amanda_wages
        state_income_weights = np.zeros(6)  # (6,)
        state_local_weights = np.zeros(6)  # (6,)
        state_income_weights[0] = ytd[0] + ytd[1]
        prior_tax = annual_tax(ytd[0] * 12 / 9, ytd[1] * 12 / 9, 0., 0., 0., 0., 0., 0., 0., False, 0, 0., "PA", 0., ytd[0] * 12 / 9, 1., ss_base=tax_settings[2], standard_single=tax_settings[3], child_credit=tax_settings[4], tax_year=2026)
        withheld = sum(prior_tax) * .75 if settings.prior_withholding < 0 else settings.prior_withholding
        ytd_components = np.zeros(3)  # (federal, state/local, payroll)
        ytd_components[:] = np.array(prior_tax) * .75
        ytd_qsbs = 0.
        prev_net = np.sum(cash) + np.sum(broker) + np.sum(retirement) - np.sum(balance)
        for m in range(t):
            year, month = int(schedule[m, SCHEDULE.year]), int(schedule[m, SCHEDULE.month])
            state_slot = int(schedule[m, SCHEDULE.state])
            state = STATE_CODES[state_slot]
            joint = schedule[m, SCHEDULE.married] > .5
            children = int(schedule[m, SCHEDULE.children])
            stopped = schedule[m, SCHEDULE.amanda_stopped] > .5
            scale = (1 + tax_settings[0]) ** (year - 2026)
            inflation *= max(.9, 1 + draws[path, m, 1])
            if schedule[m, SCHEDULE.logan_new_role] > .5:
                wage_index, pending_raise = inflation, 1.
            elif schedule[m, SCHEDULE.logan_raise_due] > .5:
                wage_index *= pending_raise
                pending_raise = 1.
            if schedule[m, SCHEDULE.amanda_raise_due] > .5:
                amanda_wage_index *= amanda_pending_raise
                amanda_pending_raise = 1.
            pending_raise *= max(.95, 1 + draws[path, m, 4])
            amanda_pending_raise *= max(.95, 1 + draws[path, m, 4])
            capital_return = draws[path, m, 0]
            return_dollars = np.sum(broker + retirement) * capital_return
            broker *= 1 + capital_return  # (2,)
            retirement *= 1 + capital_return  # (2,)
            cash_interest = cash * draws[path, m, 3] / 12  # (2,)
            cash += cash_interest  # (2,)
            # Total-return draw is split into price return and dividends, not counted twice.
            dividends = np.maximum(0., broker) * settings.dividend_yield / 12  # (2,)
            broker -= dividends  # (2,)
            cash += dividends  # (2,)
            home_index *= max(.1, 1 + draws[path, m, 2])
            home_gain = house_value * draws[path, m, 2]
            house_value += home_gain
            if month == 4:
                settlement = tax_due
                tax_due = 0.
            else:
                settlement = 0.
            for person in range(2):
                u = draws[path, m, 5] if person == 0 else (draws[path, m, 5] + .431) % 1
                job_probability = settings.ud_job_loss_probability if person == 0 and schedule[m, SCHEDULE.ud_wage] > 0 else settings.job_loss_probability
                if m >= 3 and employed_gap[person] == 0 and u < job_probability * job_multiplier / 12:
                    employed_gap[person] = int(settings.job_gap_months)
            wage1 = schedule[m, SCHEDULE.logan_wage] * wage_index if employed_gap[0] == 0 else 0.
            if schedule[m, SCHEDULE.ud_wage] == 0:
                wage1 *= salary_multipliers[path]
            darpa_fallback = schedule[m, SCHEDULE.darpa_blend] > .5 and company[path, m, BUSINESS.darpa_awarded] < .5
            if darpa_fallback:
                wage1 = schedule[m, SCHEDULE.fallback_wage] * wage_index if employed_gap[0] == 0 else 0.
            wage2 = schedule[m, SCHEDULE.amanda_salary] * amanda_wage_index if not stopped and employed_gap[1] == 0 else 0.
            if schedule[m, SCHEDULE.amanda_leave] > .5:
                wage2 *= settings.leave_pay
            if employed_gap[0] > 0:
                employed_gap[0] -= 1
            if employed_gap[1] > 0:
                employed_gap[1] -= 1
            founder_pay = company[path, m, BUSINESS.founder_pay]
            corporate = company[path, m, BUSINESS.c_corp] > .5
            business_wage = founder_pay if corporate else 0.
            blog = settings.blog_subscribers * settings.blog_net * (1 + settings.blog_growth) ** m
            se = blog + (0. if corporate else founder_pay)
            retirement1 = min(max(0., tax_settings[1] * scale - ytd[6]), (wage1 + business_wage) * persons[0, 5])
            retirement2 = min(max(0., tax_settings[1] * scale - ytd[7]), wage2 * persons[1, 5])
            match1 = wage1 * (schedule[m, SCHEDULE.fallback_match] if darpa_fallback else schedule[m, SCHEDULE.logan_match]) + business_wage * .04
            match2 = wage2 * persons[1, 6]
            benefits = (darpa_fallback or schedule[m, SCHEDULE.logan_benefits] > .5) and (wage1 > 0 or founder_pay > 0)
            if stopped:
                if benefits:
                    premium = settings.family_medical * inflation
                    if not joint and settings.domestic_partner < .5:
                        premium += settings.replacement_adult * inflation
                    pretax = min(wage1 + business_wage, settings.family_medical * inflation)
                else:
                    premium = (2 * settings.replacement_adult + children * settings.replacement_child) * inflation
                    pretax = 0.
            else:
                premium = (persons[0, 7] if benefits else settings.replacement_adult) * inflation
                premium += (persons[1, 7] if wage2 > 0 else settings.replacement_adult) * inflation
                if children:
                    premium += max(0., settings.family_medical - persons[0, 7]) * inflation if benefits else children * settings.replacement_child * inflation
                pretax = min(wage1 + business_wage, premium - persons[1, 7] * inflation) if benefits else 0.
                pretax += min(wage2, persons[1, 7] * inflation)
            childcare = 0. if stopped else children * schedule[m, SCHEDULE.childcare_price] * inflation
            child_other = children * settings.child_other * inflation
            oop = (settings.medical_oop / 12 * (1 + .5 * children) + schedule[m, SCHEDULE.new_birth] * settings.birth_cost) * inflation
            personal1 = persons[0, 4] * schedule[m, SCHEDULE.cost_factor] * inflation
            personal2 = persons[1, 4] * schedule[m, SCHEDULE.cost_factor] * inflation
            wedding = schedule[m, SCHEDULE.wedding] * settings.wedding_cost * inflation
            repairs = settings.repair_cost * inflation if draws[path, m, 6] < settings.repair_probability / 12 else 0.
            moving = schedule[m, SCHEDULE.move] * home.moving_cost * inflation
            required_debt, debt_interest = 0., 0.
            loan_principal_payment = np.zeros(len(balance))  # (d,)
            for d in range(len(balance)):
                deferred = m < int(loans[d, 3])
                if not deferred:
                    accrued_deferred[d] = 0.
                interest = (balance[d] - accrued_deferred[d]) * loans[d, 1] / 12 if not deferred or loans[d, 4] < .5 else 0.
                if deferred:
                    accrued_deferred[d] += interest
                balance[d] += interest
                debt_interest += interest
                if not deferred:
                    if minimums[d] == 0 and balance[d] > 0:
                        minimums[d] = payment(balance[d], loans[d, 1] / 12, int(loans[d, 5]))
                    loan_principal_payment[d] = min(balance[d], minimums[d])
                    required_debt += loan_principal_payment[d]
                    balance[d] -= loan_principal_payment[d]
            rent = schedule[m, SCHEDULE.rent] * inflation * (1 + children * home.bedroom_premium) if house_value == 0 else 0.
            mortgage_interest, principal_payment, property_tax = 0., 0., 0.
            if mortgage > 0:
                mortgage_interest = mortgage * home.mortgage_rate / 12
                principal_payment = min(mortgage, max(0., mortgage_pay - mortgage_interest))
                mortgage -= principal_payment
            if house_value > 0:
                property_tax = house_value * schedule[m, SCHEDULE.property_tax_rate] / 12
            maintenance = house_value * (home.maintenance + home.insurance) / 12
            pmi = mortgage * home.pmi / 12 if house_value > 0 and mortgage / house_value > .8 else 0.
            housing_cost = rent + mortgage_interest + property_tax + maintenance + pmi
            equity_cash = company[path, m, BUSINESS.equity_proceeds]
            ordinary_exit = company[path, m, BUSINESS.ordinary_proceeds]
            distribution = company[path, m, BUSINESS.distribution]
            # Equity basis recovered proportionally against cash proceeds. Unknown basis defaults to zero.
            exit_gain = max(0., equity_cash - company[path, m, BUSINESS.exit_basis_recovered])
            other_income = float(np.sum(cash_interest)) + ordinary_exit + distribution
            # Provisional pass-through tax distributions: retained profit must also be taxable.
            if not corporate:
                profit_share = max(0., company[path, m, BUSINESS.revenue] + company[path, m, BUSINESS.grant_claims] - company[path, m, BUSINESS.operating_cost]) * company[path, m, BUSINESS.founder_share]
                other_income += max(0., profit_share - distribution)
            de_source = schedule[m, SCHEDULE.delaware_source] * wage_index
            if darpa_fallback and schedule[m, SCHEDULE.logan_wage] > 0:
                de_source *= schedule[m, SCHEDULE.fallback_wage] / schedule[m, SCHEDULE.logan_wage]
            federal, state_tax, fica = annual_tax(wage1 * 12 + business_wage * 12, wage2 * 12, se * 12,
                other_income * 12, 0., np.sum(dividends) * 12, retirement1 * 12, retirement2 * 12, pretax * 12,
                joint, children, childcare * 12, state, schedule[m, SCHEDULE.local_tax], de_source * 12,
                scale, mortgage_interest * 12, property_tax * 12, ss_base=tax_settings[2], standard_single=tax_settings[3], child_credit=tax_settings[4], tax_year=year)
            federal_month, state_month, fica_month = federal / 12, state_tax / 12, fica / 12
            monthly_tax = federal_month + state_month + fica_month
            # Withhold incremental exit tax once rather than annualizing the exit.
            if equity_cash + ordinary_exit > 0:
                before = annual_tax(wage1 * 12 + business_wage * 12, wage2 * 12, se * 12, max(0., other_income - ordinary_exit) * 12,
                    0., np.sum(dividends) * 12, retirement1 * 12, retirement2 * 12, pretax * 12, joint, children, childcare * 12,
                    state, schedule[m, SCHEDULE.local_tax], de_source * 12, scale, ss_base=tax_settings[2], standard_single=tax_settings[3], child_credit=tax_settings[4], tax_year=year)
                after = annual_tax(wage1 * 12 + business_wage * 12, wage2 * 12, se * 12, max(0., other_income - ordinary_exit) * 12 + ordinary_exit,
                    exit_gain, np.sum(dividends) * 12, retirement1 * 12, retirement2 * 12, pretax * 12, joint, children, childcare * 12,
                    state, schedule[m, SCHEDULE.local_tax], de_source * 12, scale, 0., 0., company[path, m, BUSINESS.qsbs_exclusion], bool(tax_settings[6 + state_slot]), ss_base=tax_settings[2], standard_single=tax_settings[3], child_credit=tax_settings[4], tax_year=year)
                federal_month = before[0] / 12 + after[0] - before[0]
                state_month = before[1] / 12 + after[1] - before[1]
                fica_month = before[2] / 12 + after[2] - before[2]
                monthly_tax = federal_month + state_month + fica_month
            income1 = wage1 + founder_pay + blog + distribution + equity_cash + ordinary_exit
            income2 = wage2
            share1 = income1 / max(1e-12, income1 + income2) if income1 + income2 > 0 else .5
            shared = childcare + child_other + oop + premium + wedding + repairs + moving + housing_cost + principal_payment
            # Optional deferrals stop before drawing savings to meet required bills.
            available1 = cash[0] + income1 - personal1 - shared * share1 - required_debt - (monthly_tax + settlement) * share1
            available2 = cash[1] + income2 - personal2 - shared * (1-share1) - (monthly_tax + settlement) * (1-share1)
            if (available1 + available2 < retirement1 + retirement2) if joint else available1 < retirement1:
                retirement1, match1 = 0., 0.
            if (available1 + available2 < retirement1 + retirement2) if joint else available2 < retirement2:
                retirement2, match2 = 0., 0.
            # Annual reconciliation uses actual deferrals; monthly withholding is an estimate.
            cash[0] += income1 - personal1 - shared * share1 - required_debt - monthly_tax * share1 - retirement1 - settlement * share1
            cash[1] += income2 - personal2 - shared * (1-share1) - monthly_tax * (1-share1) - retirement2 - settlement * (1-share1)
            retirement[0] += retirement1 + match1
            retirement[1] += retirement2 + match2
            realized = 0.
            shortfall = 0.
            for i in range(2):
                if cash[i] < 0 and joint and cash[1-i] > 0:
                    transfer = min(-cash[i], cash[1-i])
                    cash[i] += transfer
                    cash[1-i] -= transfer
                if cash[i] < 0:
                    sale = min(-cash[i], broker[i])
                    sold_basis = basis[i] * sale / max(broker[i], 1e-12)
                    realized += sale - sold_basis
                    broker[i] -= sale
                    basis[i] -= sold_basis
                    cash[i] += sale
                if cash[i] < 0 and joint:
                    j = 1-i
                    sale = min(-cash[i], broker[j])
                    sold_basis = basis[j] * sale / max(broker[j], 1e-12)
                    realized += sale - sold_basis
                    broker[j] -= sale
                    basis[j] -= sold_basis
                    cash[i] += sale
                if cash[i] < 0:
                    shortfall -= cash[i]
                    unpaid[i] -= cash[i]
                    cash[i] = 0.
            # Repay previously unmet obligations before voluntary saving/investing.
            for i in range(2):
                paid_arrears = min(cash[i], unpaid[i])
                cash[i] -= paid_arrears
                unpaid[i] -= paid_arrears
            extra_debt, brokerage_contribution, purchase = 0., 0., 0.
            # Optional allocation uses this month's surplus, not all accumulated reserves.
            surplus1 = max(0., income1 - personal1 - shared * share1 - required_debt - monthly_tax * share1 - retirement1)
            discretionary = min(cash[0], max(0., surplus1 - settings.cash_target)) if shortfall == 0 else 0.
            if settings.cash_cap > 0 and cash[0] > settings.cash_cap:
                discretionary = min(cash[0] - settings.cash_cap, surplus1)
            highest_apr = 0.
            for d in range(len(balance)):
                if balance[d] > .01 and not (m < loans[d, 3] and loans[d, 4] > .5):
                    highest_apr = max(highest_apr, loans[d, 1])
            debt_share = min(1., max(0., .5 + (highest_apr - settings.debt_hurdle) / max(1e-9, 2 * settings.debt_transition_band))) if highest_apr > 0 else 0.
            budget = discretionary * debt_share
            for iteration in range(len(balance)):
                best, apr = -1, -1.
                for d in range(len(balance)):
                    if balance[d] > .01 and loans[d, 1] > apr and not (m < loans[d, 3] and loans[d, 4] > .5):
                        best, apr = d, loans[d, 1]
                if best < 0 or budget <= 0:
                    break
                paid = min(budget, balance[best])
                accrued_deferred[best] = max(0., accrued_deferred[best] - paid)
                balance[best] -= paid
                cash[0] -= paid
                budget -= paid
                extra_debt += paid
            investment = min(cash[0], max(0., discretionary - extra_debt))
            cash[0] -= investment
            broker[0] += investment
            basis[0] += investment
            brokerage_contribution += investment
            # Amanda keeps six months of her own baseline spending as a cash reserve.
            investment2 = min(max(0., cash[1] - personal2 * 6), max(0., income2 - personal2 - shared*(1-share1)-monthly_tax*(1-share1)-retirement2)) if shortfall == 0 else 0.
            cash[1] -= investment2
            broker[1] += investment2
            basis[1] += investment2
            brokerage_contribution += investment2
            closing_expense = 0.
            if schedule[m, SCHEDULE.purchase_requested] > .5 and house_value == 0:
                price = schedule[m, SCHEDULE.home_price] * home_index * (1 + children * home.bedroom_premium)
                down = price * home.down_payment
                closing = price * home.closing_cost
                new_mortgage = price - down
                required_payment = payment(new_mortgage, home.mortgage_rate / 12, int(home.mortgage_months))
                dti = (required_payment + price * (schedule[m, SCHEDULE.property_tax_rate] + home.insurance) / 12 + required_debt) / max(1., income1 + income2 - equity_cash - ordinary_exit - distribution)
                reserve = home.reserve_months * (personal1 + personal2 + shared - rent + required_payment)
                # Purchase uses cash only. Brokerage can be earmarked explicitly via a rerun.
                accessible_cash = np.sum(cash) if joint else cash[0]
                if accessible_cash >= down + closing + reserve and dti <= home.max_dti:
                    if joint:
                        fraction = cash[0] / max(1., np.sum(cash))
                        cash[0] -= (down + closing) * fraction
                        cash[1] -= (down + closing) * (1-fraction)
                    else:
                        cash[0] -= down + closing
                    house_value, mortgage, mortgage_pay = price, new_mortgage, required_payment
                    closing_expense, purchase = closing, 1.
            ytd += np.array([wage1 + business_wage, wage2, se, other_income, exit_gain + realized,
                             np.sum(dividends), retirement1, retirement2, pretax, childcare,
                             de_source, mortgage_interest, property_tax])  # (13,)
            ytd_qsbs += company[path, m, BUSINESS.qsbs_exclusion]
            state_weight = max(0., wage1 + founder_pay + wage2 + other_income + exit_gain + realized + np.sum(dividends))
            state_income_weights[state_slot] += state_weight
            state_local_weights[state_slot] += state_weight * schedule[m, SCHEDULE.local_tax]
            withheld += monthly_tax
            ytd_components += np.array([federal_month, state_month, fica_month])
            adjustment = 0.
            if month == 12:
                fed_total, state_total, fica_total = 0., 0., 0.
                for state_index in range(6):
                    weight = state_income_weights[state_index] / max(1., np.sum(state_income_weights))
                    if weight == 0:
                        continue
                    fed, state_part, payroll = annual_tax(ytd[0], ytd[1], ytd[2], ytd[3], ytd[4], ytd[5], ytd[6], ytd[7], ytd[8], joint, children,
                        ytd[9], STATE_CODES[state_index], state_local_weights[state_index] / max(1., state_income_weights[state_index]), ytd[10], scale,
                        ytd[11], ytd[12], ytd_qsbs, bool(tax_settings[6 + state_index]), ss_base=tax_settings[2], standard_single=tax_settings[3], child_credit=tax_settings[4], tax_year=year)
                    fed_total += fed * weight
                    state_total += state_part * weight
                    fica_total += payroll * weight
                federal_month += fed_total - ytd_components[0]
                state_month += state_total - ytd_components[1]
                fica_month += fica_total - ytd_components[2]
                ytd_components[:] = 0.
                adjustment = fed_total + state_total + fica_total - withheld
                tax_due += adjustment
                ytd[:] = 0.  # (13,)
                state_income_weights[:] = 0.  # (6,)
                state_local_weights[:] = 0.  # (6,)
                withheld, ytd_qsbs = 0., 0.
            tax_expense = monthly_tax + adjustment
            financial = np.sum(cash + broker + retirement) - np.sum(balance) - tax_due - np.sum(unpaid)
            net = financial + house_value - mortgage
            expected_change = income1 + income2 + np.sum(cash_interest) + return_dollars + home_gain + match1 + match2 - (personal1 + personal2 + shared - principal_payment + debt_interest + tax_expense + closing_expense)
            residual = net - prev_net - expected_change
            prev_net = net
            output[path, m] = np.array([np.sum(cash), np.sum(broker), np.sum(basis), np.sum(retirement), np.sum(balance), house_value,
                mortgage, tax_due, np.sum(unpaid), np.sum(cash + broker) - tax_due - np.sum(unpaid), financial, net, company[path, m, BUSINESS.company_value],
                wage1, wage2, founder_pay, personal1 + personal2, childcare, child_other, premium, housing_cost, tax_expense,
                federal_month, state_month, fica_month, retirement1+retirement2, match1+match2, required_debt+extra_debt, debt_interest,
                np.sum(cash_interest), realized, equity_cash+ordinary_exit, shortfall, purchase, inflation, residual,
                cash[0], cash[1], retirement[0], retirement[1], retirement2, match2, children, (1.0 if stopped else 0.0), (1.0 if joint else 0.0),
                wage1+wage2+founder_pay+blog, return_dollars, house_value-mortgage, settlement, mortgage_interest, property_tax,
                brokerage_contribution, extra_debt, distribution, principal_payment+mortgage_interest, oop])  # (k,)
    return output  # (p,t,k)


def simulate(config: RunConfig, scenario: Scenario, draws: np.ndarray | None = None,
             forecast: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return household (p,t,k) and company (p,t,b) paths including the 2026 bridge."""
    if draws is None:
        draws = economic_paths(config, scenario.macro)  # (p,t,8)
    schedule = build_schedule(config, scenario)  # (t,26)
    company = simulate_business(config, scenario, schedule, draws, forecast)  # (p,t,21)
    h, housing = config.household, config.housing
    persons = np.asarray([[v.cash, v.brokerage, v.brokerage_basis, v.retirement, v.monthly_personal_spend,
                           v.retirement_rate, v.employer_match, v.medical_monthly, v.age] for v in (h.logan, h.amanda)], dtype=float)  # (2,9)
    if not h.amanda_retirement_enabled:
        persons[1, 5:7] = 0  # (2,)
    loan_rows = []
    for loan in h.loans:
        # Interest/required payments change in the first full month after deferment.
        until = np.datetime64(loan.deferred_until, "M")
        offset = max(0, int((until - np.datetime64("2026-10", "M")).astype(int)) + 1)
        loan_rows.append([loan.balance, loan.apr, loan.minimum, offset, float(loan.subsidized), loan.repayment_months])
    loans = np.asarray(loan_rows, dtype=float).reshape(-1, 6)  # (d,6)
    settings = HouseholdParameters(h.cash_target_monthly, h.cash_cap or 0, h.debt_hurdle,
                           h.family_medical_monthly, h.replacement_adult_medical, h.replacement_child_medical,
                           float(h.domestic_partner_eligible), h.child_monthly_other, h.parental_leave_pay_fraction,
                           h.medical_annual_oop, h.birth_cost, h.wedding_cost, h.job_loss_annual_probability,
                           h.job_gap_months, h.repair_annual_probability, h.repair_cost, config.economics.dividend_yield,
                           h.blog_subscribers, h.blog_net_per_subscriber, h.logan_2026_prior_wages,
                           h.amanda_2026_prior_wages, h.prior_2026_withholding if h.prior_2026_withholding is not None else -1,
                           h.blog_monthly_growth, h.debt_transition_band, h.ud_job_loss_annual_probability)
    home = HousingParameters(housing.down_payment, housing.closing_cost, housing.sale_cost, housing.mortgage_rate,
                       housing.mortgage_years*12, housing.reserve_months, housing.maintenance_rate,
                       housing.insurance_rate, housing.pmi_rate, housing.max_dti,
                       housing.bedroom_premium_per_child, housing.moving_cost)
    taxes = np.asarray([config.taxes.index_rate, config.taxes.employee_deferral_limit,
                        config.taxes.social_security_base, config.taxes.federal_standard_single,
                        config.taxes.child_credit, config.business.equity_basis,
                        *[float(state in config.taxes.qsbs_state_conformity) for state in STATES]], dtype=float)  # (12,)
    settings = HouseholdParameters._make(map(float, settings))
    home = HousingParameters._make(map(float, home))
    anchor = config.career_salary_overrides.get(scenario.career, CAREER_SALARY[scenario.career])
    salary_multipliers = starting_salaries(config, scenario.career) / anchor if anchor > 0 else np.ones(config.paths)  # (paths,)
    household = household_kernel(schedule, draws, company, persons, loans, settings, home, taxes, MACROS[scenario.macro][2], salary_multipliers)  # (p,t,k)
    error = np.max(np.abs(household[:, :, METRICS.index("reconciliation_error")]))
    if error > .05:
        raise ArithmeticError(f"Household ledger reconciliation failed: ${error:.6f}")
    return household, company
