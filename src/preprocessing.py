from __future__ import annotations

import pandas as pd

TARGET_COLUMNS = ("GDS", "GDS_R1", "GDS_R2", "GDS_R3", "GDS_R4", "GDS_R5")
EXCLUDE_COLUMNS = ("ID",) + TARGET_COLUMNS


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DataFrame con solo las columnas de características, excluyendo ID y
    las columnas objetivo."""
    return df.drop(columns=list(EXCLUDE_COLUMNS), errors="ignore")


def split_features_target(
    df: pd.DataFrame, target_col: str = "GDS"
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa el DataFrame en un DataFrame de características (X) y una Serie de objetivo (y).
    Por defecto, se asume que la columna objetivo es 'GDS'."""
    x_frame = feature_frame(df)
    y_series = df[target_col]
    return x_frame, y_series


def validate_binary_features(x_frame: pd.DataFrame) -> None:
    """Valida que todas las columnas de características contengan solo valores binarios (0 o 1)."""
    if not x_frame.isin([0, 1]).all().all():
        raise ValueError(
            "Algunas columnas de características contienen valores no binarios."
        )
