from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from .balancing import oversample_minority_classes

ModelBuilder = Callable[[int], ClassifierMixin]
SplitStrategy = Literal["stratified_kfold", "loocv"]


def choose_n_splits(y_series: pd.Series, requested_splits: int = 5) -> int:
    min_class_count = int(y_series.value_counts().min())
    if min_class_count < 2:
        raise ValueError(
            "Cross-validation requires at least 2 samples in every class. "
            f"Minimum class count found: {min_class_count}."
        )
    return min(requested_splits, min_class_count)


def cross_validated_predictions(
    x_frame: pd.DataFrame,
    y_series: pd.Series,
    model_builder: ModelBuilder,
    strategy: SplitStrategy = "stratified_kfold",
    requested_splits: int = 5,
    random_state: int = 42,
    apply_oversampling: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    x_values = x_frame.to_numpy()
    y_values = y_series.to_numpy()

    if strategy == "stratified_kfold":
        effective_splits = choose_n_splits(y_series, requested_splits=requested_splits)
        splitter = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        split_iterator = splitter.split(x_values, y_values)
    elif strategy == "loocv":
        effective_splits = len(y_values)
        split_iterator = LeaveOneOut().split(x_values, y_values)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    predictions = np.empty(len(y_values), dtype=object)
    filled_mask = np.zeros(len(y_values), dtype=bool)

    for fold_idx, (train_idx, test_idx) in enumerate(split_iterator, start=1):
        x_train = x_values[train_idx]
        y_train = y_values[train_idx]
        x_test = x_values[test_idx]

        if apply_oversampling:
            x_train, y_train = oversample_minority_classes(
                x_train,
                y_train,
                random_state=random_state + fold_idx,
            )

        model = model_builder(random_state)
        model.fit(x_train, y_train)
        fold_predictions = model.predict(x_test)
        predictions[test_idx] = fold_predictions.tolist()
        filled_mask[test_idx] = True

    if not bool(filled_mask.all()):
        raise RuntimeError("Cross-validation did not produce predictions for all samples.")

    y_pred = np.asarray(predictions)
    if y_values.dtype != object:
        y_pred = y_pred.astype(y_values.dtype, copy=False)

    return y_values, y_pred, effective_splits


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_model(
    x_frame: pd.DataFrame,
    y_series: pd.Series,
    model_builder: ModelBuilder,
    strategy: SplitStrategy = "stratified_kfold",
    requested_splits: int = 5,
    random_state: int = 42,
    apply_oversampling: bool = True,
) -> tuple[dict[str, float], np.ndarray, list, int]:
    y_true, y_pred, effective_splits = cross_validated_predictions(
        x_frame=x_frame,
        y_series=y_series,
        model_builder=model_builder,
        strategy=strategy,
        requested_splits=requested_splits,
        random_state=random_state,
        apply_oversampling=apply_oversampling,
    )

    labels = list(pd.Series(y_series).dropna().sort_values().unique())
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = metrics_from_predictions(y_true, y_pred)
    return metrics, conf_matrix, labels, effective_splits


def run_model_suite(
    x_frame: pd.DataFrame,
    y_series: pd.Series,
    model_builders: dict[str, ModelBuilder],
    strategy: SplitStrategy = "stratified_kfold",
    requested_splits: int = 5,
    random_state: int = 42,
    apply_oversampling: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray | list]], int]:
    metrics_rows: list[dict[str, float | str]] = []
    confusion_by_model: dict[str, dict[str, np.ndarray | list]] = {}
    effective_splits: int | None = None

    for model_name, model_builder in model_builders.items():
        metrics, conf_matrix, labels, current_splits = evaluate_model(
            x_frame=x_frame,
            y_series=y_series,
            model_builder=model_builder,
            strategy=strategy,
            requested_splits=requested_splits,
            random_state=random_state,
            apply_oversampling=apply_oversampling,
        )

        if effective_splits is None:
            effective_splits = current_splits
        elif effective_splits != current_splits:
            raise RuntimeError("Inconsistent fold count across models during evaluation.")

        metrics_rows.append({"model": model_name, **metrics})
        confusion_by_model[model_name] = {
            "matrix": conf_matrix,
            "labels": labels,
        }

    metrics_df = pd.DataFrame(metrics_rows).set_index("model")
    metrics_df = metrics_df.sort_values("f1_macro", ascending=False)
    if effective_splits is None:
        raise RuntimeError("No models were evaluated.")
    return metrics_df, confusion_by_model, effective_splits
