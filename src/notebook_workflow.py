from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import math
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    SequentialFeatureSelector,
    VarianceThreshold,
    chi2,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import TARGET_COLUMNS, feature_frame

ValidationStrategy = Literal["stratified_kfold", "loocv"]
CODIFICATION_COLUMNS = TARGET_COLUMNS


class NaiveBayesManual:
    def fit(self, x_values: np.ndarray, y_values: np.ndarray) -> None:
        self.classes = set(y_values)
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.total = len(y_values)

        for row, label in zip(x_values, y_values):
            self.class_counts[label] += 1
            for idx, value in enumerate(row):
                self.feature_counts[label][(idx, value)] += 1

    def predict_one(self, row: np.ndarray):
        scores = {}
        for class_label in self.classes:
            class_prior = self.class_counts[class_label] / self.total
            log_prob = math.log(class_prior)
            for idx, value in enumerate(row):
                freq = self.feature_counts[class_label][(idx, value)] + 1
                total = self.class_counts[class_label] + 2
                log_prob += math.log(freq / total)
            scores[class_label] = log_prob
        return max(scores, key=scores.get)

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        predictions = [self.predict_one(row) for row in x_values]
        return np.asarray(predictions, dtype=object)


def choose_n_splits(y_series: pd.Series, requested_splits: int = 5) -> int:
    """Determina el número efectivo de splits para validación cruzada, asegurando
    que cada clase tenga al menos 2 muestras en cada fold."""
    min_class_count = int(y_series.value_counts().min())
    if min_class_count < 2:
        raise ValueError(
            "Cross-validation requiere al menos 2 muestras por clase."
            f"Cantidad de muestras en la clase minoritaria: {min_class_count}."
        )
    return min(requested_splits, min_class_count)


def build_codification_distributions(
    df: pd.DataFrame,
    codification_columns: Iterable[str] = CODIFICATION_COLUMNS,
) -> dict[str, pd.DataFrame]:
    """Construye un diccionario de distribuciones de clases para cada columna de codificación."""
    distributions: dict[str, pd.DataFrame] = {}
    for column in codification_columns:
        if column not in df.columns:
            continue
        counts = df[column].value_counts().sort_index()
        distribution = counts.rename("count").to_frame()
        distribution["ratio"] = (
            distribution["count"] / distribution["count"].sum()
        ).round(6)
        distribution.index.name = "class"
        distributions[column] = distribution
    return distributions


def run_feature_selection_study(
    df: pd.DataFrame,
    target_col: str = "GDS",
    random_state: int = 42,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Ejecuta un estudio de selección de características utilizando
    varias técnicas y devuelve los resultados en un formato estructurado."""
    x_frame = feature_frame(df)
    y_series = df[target_col]

    valid_rows = y_series.notna()
    x_frame = x_frame.loc[valid_rows]
    y_series = y_series.loc[valid_rows]

    variance_selector = VarianceThreshold(threshold=0.02)
    variance_selector.fit(x_frame)
    variance_selected = pd.DataFrame(
        {"feature": x_frame.columns[variance_selector.get_support()]}
    )

    top_k = min(15, x_frame.shape[1])
    chi2_top_selector = SelectKBest(score_func=chi2, k=top_k)
    chi2_top_selector.fit(x_frame, y_series)
    top_mask = chi2_top_selector.get_support()
    chi2_top = pd.DataFrame(
        {
            "feature": x_frame.columns[top_mask],
            "chi2_score": chi2_top_selector.scores_[top_mask],
        }
    ).sort_values("chi2_score", ascending=False)

    chi2_all_selector = SelectKBest(score_func=chi2, k="all")
    chi2_all_selector.fit(x_frame, y_series)
    chi2_all = pd.DataFrame(
        {
            "feature": x_frame.columns,
            "chi2_score": chi2_all_selector.scores_,
            "p_value": chi2_all_selector.pvalues_,
        }
    ).sort_values("chi2_score", ascending=False)

    rfe_features = min(10, x_frame.shape[1])

    logistic_model = LogisticRegression(max_iter=10000, random_state=random_state)
    rfe_logistic = RFE(estimator=logistic_model, n_features_to_select=rfe_features)
    rfe_logistic.fit(x_frame, y_series)
    rfe_logistic_df = pd.DataFrame(
        {
            "feature": x_frame.columns,
            "ranking": rfe_logistic.ranking_,
            "selected": rfe_logistic.support_,
        }
    ).sort_values(["ranking", "feature"])

    tree_model = DecisionTreeClassifier(random_state=random_state)
    rfe_tree = RFE(estimator=tree_model, n_features_to_select=rfe_features)
    rfe_tree.fit(x_frame, y_series)
    rfe_tree_df = pd.DataFrame(
        {
            "feature": x_frame.columns,
            "ranking": rfe_tree.ranking_,
            "selected": rfe_tree.support_,
        }
    ).sort_values(["ranking", "feature"])

    forward_features = min(5, x_frame.shape[1])
    forward_cv = choose_n_splits(y_series, requested_splits=5)

    knn_model = KNeighborsClassifier(n_neighbors=3)
    forward_selector = SequentialFeatureSelector(
        estimator=knn_model,
        n_features_to_select=forward_features,
        direction="forward",
        scoring="accuracy",
        cv=forward_cv,
        n_jobs=-1,
    )
    forward_pipeline = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("selector", forward_selector),
        ]
    )
    forward_pipeline.fit(x_frame, y_series)
    forward_selected = x_frame.columns[
        forward_pipeline.named_steps["selector"].get_support()
    ]
    forward_df = pd.DataFrame({"feature": forward_selected})

    outputs = {
        "variance_selected": variance_selected,
        "chi2_top15": chi2_top,
        "chi2_all": chi2_all,
        "rfe_logistic": rfe_logistic_df,
        "rfe_tree": rfe_tree_df,
        "forward_knn": forward_df,
    }
    metadata = {
        "top_k": top_k,
        "rfe_features": rfe_features,
        "forward_features": forward_features,
        "forward_cv": forward_cv,
    }
    return outputs, metadata


def run_pca_study(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Ejecuta un estudio de PCA para analizar la varianza explicada y reducir
    la dimensionalidad manteniendo el 95% de la varianza."""
    x_frame = feature_frame(df)

    pca_full = PCA(n_components=None)
    pca_full.fit(x_frame)

    explained_df = pd.DataFrame(
        {
            "component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca_full.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(
                pca_full.explained_variance_ratio_
            ),
        }
    )

    pca_95 = PCA(n_components=0.95)
    x_pca_95 = pca_95.fit_transform(x_frame)
    pca_95_df = pd.DataFrame(
        x_pca_95, columns=[f"PC{i + 1}" for i in range(pca_95.n_components_)]
    )

    return explained_df, pca_95_df, int(pca_95.n_components_)


def _chi2_select_features(
    x_frame: pd.DataFrame,
    y_series: pd.Series,
    k_best: int,
) -> tuple[SelectKBest, list[str]]:
    """Realiza selección de características utilizando chi2 y devuelve
    el selector ajustado"""
    k_value = min(k_best, x_frame.shape[1])
    selector = SelectKBest(score_func=chi2, k=k_value)
    selector.fit(x_frame, y_series)
    selected_columns = list(x_frame.columns[selector.get_support()])
    return selector, selected_columns


def _predict_with_stratified_kfold(
    x_frame: pd.DataFrame,
    y_series: pd.Series,
    requested_splits: int,
    random_state: int,
    k_best: int,
) -> tuple[np.ndarray, int]:
    """Realiza predicciones utilizando Stratified K-Fold Cross-Validation con
    selección de características chi2."""
    n_splits = choose_n_splits(y_series, requested_splits=requested_splits)

    x_values = x_frame.to_numpy()
    y_values = y_series.to_numpy()

    predictions = np.empty(len(y_values), dtype=object)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    for train_idx, test_idx in splitter.split(x_values, y_values):
        x_train_frame = x_frame.iloc[train_idx]
        y_train = y_series.iloc[train_idx]
        x_test_frame = x_frame.iloc[test_idx]

        selector, _ = _chi2_select_features(
            x_frame=x_train_frame,
            y_series=y_train,
            k_best=k_best,
        )
        x_train_selected = selector.transform(x_train_frame)
        x_test_selected = selector.transform(x_test_frame)

        model = NaiveBayesManual()
        model.fit(x_train_selected, y_train.to_numpy())
        fold_predictions = model.predict(x_test_selected)
        predictions[test_idx] = fold_predictions.tolist()

    y_pred = np.asarray(predictions)
    if y_values.dtype != object:
        y_pred = y_pred.astype(y_values.dtype, copy=False)

    return y_pred, n_splits


def _predict_with_loocv(
    x_frame: pd.DataFrame,
    y_series: pd.Series,
    k_best: int,
) -> tuple[np.ndarray, int]:
    """Realiza predicciones utilizando Leave-One-Out Cross-Validation (LOOCV) con
    selección de características chi2."""
    x_values = x_frame.to_numpy()
    y_values = y_series.to_numpy()

    predictions = np.empty(len(y_values), dtype=object)

    splitter = LeaveOneOut()
    for train_idx, test_idx in splitter.split(x_values, y_values):
        x_train_frame = x_frame.iloc[train_idx]
        y_train = y_series.iloc[train_idx]
        x_test_frame = x_frame.iloc[test_idx]

        selector, _ = _chi2_select_features(
            x_frame=x_train_frame,
            y_series=y_train,
            k_best=k_best,
        )
        x_train_selected = selector.transform(x_train_frame)
        x_test_selected = selector.transform(x_test_frame)

        model = NaiveBayesManual()
        model.fit(x_train_selected, y_train.to_numpy())
        fold_predictions = model.predict(x_test_selected)
        predictions[test_idx] = fold_predictions.tolist()

    y_pred = np.asarray(predictions)
    if y_values.dtype != object:
        y_pred = y_pred.astype(y_values.dtype, copy=False)

    return y_pred, len(y_values)


def _evaluate_single_codification(
    x_frame: pd.DataFrame,
    y_series: pd.Series,
    strategy: ValidationStrategy,
    requested_splits: int,
    random_state: int,
    k_best: int,
) -> tuple[dict[str, float | int], list[str]]:
    """Evalúa un modelo Naive Bayes manual utilizando la codificación especificada y devuelve las métricas
    y las columnas seleccionadas por chi2."""
    _, selected_columns = _chi2_select_features(
        x_frame=x_frame, y_series=y_series, k_best=k_best
    )
    y_values = y_series.to_numpy()

    if strategy == "stratified_kfold":
        y_pred, cv_splits = _predict_with_stratified_kfold(
            x_frame=x_frame,
            y_series=y_series,
            requested_splits=requested_splits,
            random_state=random_state,
            k_best=k_best,
        )
    elif strategy == "loocv":
        y_pred, cv_splits = _predict_with_loocv(
            x_frame=x_frame,
            y_series=y_series,
            k_best=k_best,
        )
    else:
        raise ValueError(f"Estrategia de validación desconocida: {strategy}")

    metrics = {
        "accuracy": float(accuracy_score(y_values, y_pred)),
        "precision_macro": float(
            precision_score(y_values, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_values, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_values, y_pred, average="macro", zero_division=0)),
        "selected_feature_count": len(selected_columns),
        "cv_splits": int(cv_splits),
    }
    return metrics, selected_columns


def evaluate_naive_bayes_codifications(
    df: pd.DataFrame,
    strategy: ValidationStrategy,
    requested_splits: int = 5,
    random_state: int = 42,
    k_best: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa el rendimiento de un modelo Naive Bayes manual utilizando diferentes columnas
    de codificación"""
    rows: list[dict[str, float | int | str]] = []
    selected_rows: list[dict[str, str]] = []

    for codification in CODIFICATION_COLUMNS:
        if codification not in df.columns:
            continue

        y_series = df[codification]
        valid_rows = y_series.notna()
        y_series = y_series.loc[valid_rows]
        x_frame = feature_frame(df.loc[valid_rows])

        metrics, selected_columns = _evaluate_single_codification(
            x_frame=x_frame,
            y_series=y_series,
            strategy=strategy,
            requested_splits=requested_splits,
            random_state=random_state,
            k_best=k_best,
        )
        rows.append({"codification": codification, **metrics})

        for feature_name in selected_columns:
            selected_rows.append(
                {"codification": codification, "feature": feature_name}
            )

    metrics_df = pd.DataFrame(rows).set_index("codification")
    selected_df = pd.DataFrame(selected_rows)
    return metrics_df, selected_df
