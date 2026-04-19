from __future__ import annotations

import pandas as pd

TARGET_COLUMNS = ("GDS", "GDS_R1", "GDS_R2", "GDS_R3", "GDS_R4", "GDS_R5")
EXCLUDE_COLUMNS = ("ID",) + TARGET_COLUMNS


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=list(EXCLUDE_COLUMNS), errors="ignore")


def split_features_target(df: pd.DataFrame, target_col: str = "GDS") -> tuple[pd.DataFrame, pd.Series]:
    x_frame = feature_frame(df)
    y_series = df[target_col]
    return x_frame, y_series


def validate_binary_features(x_frame: pd.DataFrame) -> None:
    if not x_frame.isin([0, 1]).all().all():
        raise ValueError("Some feature columns contain non-binary values.")
