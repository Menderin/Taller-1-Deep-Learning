from __future__ import annotations

import numpy as np
import pandas as pd


def class_distribution(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    counts = df[target_col].value_counts().sort_index()
    ratios = df[target_col].value_counts(normalize=True).sort_index()
    return pd.DataFrame({"count": counts, "ratio": ratios})


def oversample_minority_classes(
    x_values: np.ndarray,
    y_values: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    labels, counts = np.unique(y_values, return_counts=True)
    max_count = int(counts.max())
    rng = np.random.default_rng(random_state)

    sampled_indices: list[np.ndarray] = []
    for label, count in zip(labels, counts):
        class_indices = np.where(y_values == label)[0]
        if int(count) < max_count:
            extra_indices = rng.choice(class_indices, size=max_count - int(count), replace=True)
            class_indices = np.concatenate([class_indices, extra_indices])
        sampled_indices.append(class_indices)

    balanced_indices = np.concatenate(sampled_indices)
    rng.shuffle(balanced_indices)

    return x_values[balanced_indices], y_values[balanced_indices]
