"""Presentation conversion only; taxes and reconciliation use nominal ledgers."""

import pandas as pd


DIMENSIONLESS = {"date", "inflation_index", "children", "amanda_stopped", "married", "purchased",
                 "business_founder_share", "business_c_corp", "business_staff_fte", "business_darpa_awarded"}


def real_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    converted = ledger.copy()
    for column in converted:
        if column not in DIMENSIONLESS:
            converted[column] = converted[column] / ledger.inflation_index
    return converted
