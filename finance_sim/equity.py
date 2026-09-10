"""Cap-table arithmetic, preferred payouts, and conditional QSBS calculations."""

import numpy as np

from numba import njit


@njit(cache=True)
def dilute(founder: float, investor_shares: np.ndarray, amount: float,
           pre_money: float, pool_topup: float) -> tuple:
    # investor_shares: (r,); pool_topup is a new pre-money pool fraction.
    new_fraction = amount / (pre_money + amount)
    retained = (1 - pool_topup) * (1 - new_fraction)
    shares = investor_shares * retained  # (r,)
    return founder * retained, shares, new_fraction


@njit(cache=True)
def exit_waterfall(enterprise_value: float, cash: float, debt: float, fee_rate: float,
                   founder_share: float, shares: np.ndarray, invested: np.ndarray,
                   multiples: np.ndarray, participating: np.ndarray,
                   advisor_fraction: float) -> tuple:
    # shares, invested, multiples, participating: (r,), oldest round first.
    available = max(0.0, enterprise_value + cash - debt - enterprise_value * fee_rate)
    n = len(shares)
    best_founder = 0.0
    best_investors = np.zeros(n)  # (r,)
    # Enumerate preferred conversion choices, find an individually stable election.
    # At most 8 configured rounds; senior-most round is paid first.
    for mask in range(1 << n):
        remaining = available
        payouts = np.zeros(n)  # (r,)
        common_fraction = max(0.0, 1 - np.sum(shares))
        for i in range(n - 1, -1, -1):
            convert = (mask >> i) & 1
            if not convert:
                payouts[i] = min(remaining, invested[i] * multiples[i])
                remaining -= payouts[i]
            if convert or participating[i] > .5:
                common_fraction += shares[i]
        for i in range(n):
            if (mask >> i) & 1 or participating[i] > .5:
                payouts[i] += remaining * shares[i] / max(common_fraction, 1e-12)
        stable = True
        for j in range(n):
            if participating[j] > .5:
                continue
            alternate = mask ^ (1 << j)
            alternative_remaining = available
            alternative_common = max(0.0, 1 - np.sum(shares))
            alternative_pay = 0.0
            for i in range(n - 1, -1, -1):
                if not ((alternate >> i) & 1):
                    payment = min(alternative_remaining, invested[i] * multiples[i])
                    alternative_remaining -= payment
                    if i == j:
                        alternative_pay += payment
                if (alternate >> i) & 1 or participating[i] > .5:
                    alternative_common += shares[i]
            if (alternate >> j) & 1:
                alternative_pay += alternative_remaining * shares[j] / max(alternative_common, 1e-12)
            if alternative_pay > payouts[j] + 1e-6:
                stable = False
        if stable:
            ordinary_residual = remaining * max(0.0, 1 - np.sum(shares)) / max(common_fraction, 1e-12)
            advisor = ordinary_residual * advisor_fraction
            founder = (ordinary_residual - advisor) * founder_share / max(1e-12, 1 - np.sum(shares))
            return founder, advisor, payouts, available
        best_founder = remaining * founder_share / max(common_fraction, 1e-12)
        best_investors = payouts  # (r,)
    # Unusual conflicting preferences must be rejected rather than silently priced.
    return -1.0, 0.0, best_investors, available


@njit(cache=True)
def qsbs_exclusion(gain: float, original_basis: float, conversion_value: float,
                   holding_months: int, eligible: bool) -> float:
    if not eligible or holding_months < 36:
        return 0.0
    fraction = .5 if holding_months < 48 else .75 if holding_months < 60 else 1.0
    eligible_gain = max(0.0, gain - max(0.0, conversion_value - original_basis))
    return min(eligible_gain, max(15_000_000, 10 * original_basis)) * fraction
