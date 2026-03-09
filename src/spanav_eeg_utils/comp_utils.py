"""
    Title: Computational utilities

    Author: Sophie Caroni
    Date of creation: 09.03.2026

    Description:
    This script contains helper functions for computations.
"""
import pandas as pd


def fix_std_singleton(df: pd.DataFrame, std_cols: list[str], n_col: str) -> None:
    for std_col in std_cols:
        df.loc[df[n_col] == 1, std_col] = 0.0  # replace with zeros the NaNs appearing as std (if there was only one row to average across)
