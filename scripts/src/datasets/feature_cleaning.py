import numpy as np
import pandas as pd

METADATA_COLS = ["SMILES", "SELFIES", "Mols"]


def cleanFeatureDF(
    df: pd.DataFrame,
    metadata_cols: tuple[str, ...] = ("SMILES", "SELFIES", "Mols"),
    max_nan_fraction: float = 0.10,
    coerce_numeric: bool = True,
    drop_constant_cols: bool = True,
    median_impute: bool = True,
    correlation_threshold: float | None = None,
):
    df = df.copy()

    report = {
        "dropped_metadata": [],
        "dropped_high_nan_cols": [],
        "dropped_constant_cols": [],
        "median_imputed_cols": [],
        "dropped_correlated_cols": [],
    }

    meta = [c for c in metadata_cols if c in df.columns]
    if meta:
        df = df.drop(columns=meta)
        report["dropped_metadata"] = meta

    if coerce_numeric:
        df = df.apply(pd.to_numeric, errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)

    nan_frac = df.isna().mean(axis=0)
    high_nan_cols = nan_frac[nan_frac > max_nan_fraction].index.tolist()

    if high_nan_cols:
        df = df.drop(columns=high_nan_cols)
        report["dropped_high_nan_cols"] = high_nan_cols

    if drop_constant_cols:
        constant_cols = df.columns[df.nunique(dropna=True) <= 1].tolist()
        if constant_cols:
            df = df.drop(columns=constant_cols)
            report["dropped_constant_cols"] = constant_cols

    if median_impute:
        cols_with_nan = df.columns[df.isna().any(axis=0)].tolist()
        medians = df.median(axis=0, numeric_only=True)
        df = df.fillna(medians)
        report["median_imputed_cols"] = cols_with_nan

    if correlation_threshold is not None:
        corr = df.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        correlated_cols = [
            col for col in upper.columns if (upper[col] > correlation_threshold).any()
        ]

        if correlated_cols:
            df = df.drop(columns=correlated_cols)
            report["dropped_correlated_cols"] = correlated_cols

    return df, report
