import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter

import pandas as pd


if __package__ is None or __package__ == "":
    # Allow direct execution: python src/main.py
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.bagging_model import build_bagging_model
from src.boosting_model import build_boosting_model
from src.config import (
    DEFAULT_DATASET_FILENAME,
    FIGURES_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    TABLES_DIR,
    ensure_project_dirs,
)
from src.data_loader import load_spss_dataframe
from src.evaluation import run_model_suite
from src.notebook_workflow import (
    CODIFICATION_COLUMNS,
    build_codification_distributions,
    evaluate_naive_bayes_codifications,
    run_feature_selection_study,
    run_pca_study,
)
from src.preprocessing import (
    feature_frame,
    split_features_target,
    validate_binary_features,
)
from src.stacking_model import build_stacking_model
from src.visualization import (
    plot_confusion_matrix,
    plot_grouped_class_distributions,
    plot_metrics,
    plot_pca_cumulative_variance,
)

DATASET_FILENAME = DEFAULT_DATASET_FILENAME
TARGET_COLUMN = "GDS"
DESIRED_CV_SPLITS = 5
RANDOM_STATE = 42
CHI2_K_BEST = 10
APPLY_OVERSAMPLING = True

STRATEGY_DEFINITIONS: dict[str, dict[str, str]] = {
    "stratified_kfold": {
        "label": "CV Estratificado (hasta 5 folds)",
        "estimate": "~1-3 minutos",
    },
    "loocv": {
        "label": "LOOCV (Leave-One-Out)",
        "estimate": "~45-180 minutos",
    },
}

MENU_OPTIONS: dict[str, list[str]] = {
    "1": ["stratified_kfold"],
    "2": ["loocv"],
    "3": ["stratified_kfold", "loocv"],
}


def model_builders():
    """Devuelve un diccionario con los constructores de modelos de ensemble disponibles."""
    return {
        "bagging": build_bagging_model,
        "boosting": build_boosting_model,
        "stacking": build_stacking_model,
    }


def relative_path(path: Path) -> str:
    """Convierte una ruta absoluta a una ruta relativa respecto al directorio raíz del proyecto.
    Si la ruta no está dentro del proyecto, devuelve la ruta original como cadena."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def choose_execution_strategies() -> list[str]:
    """Permite al usuario seleccionar las estrategias de validación a ejecutar.
    Si la ejecución no es interactiva, se selecciona la opción 1 (CV Estratificado) por defecto.
    """
    if not sys.stdin.isatty():
        print(
            "Ejecución no interactiva detectada. Seleccionando opción 1 (CV Estratificado)."
        )
        return MENU_OPTIONS["1"]

    print("Seleccione la estrategia de validación a ejecutar:")
    print(
        f"1) CV Estratificado (hasta 5 folds) [{STRATEGY_DEFINITIONS['stratified_kfold']['estimate']}]"
    )
    print(f"2) LOOCV [{STRATEGY_DEFINITIONS['loocv']['estimate']}]")
    print("3) Ambas estrategias")

    default_option = "1"
    try:
        choice = input(f"Elige una opción [{default_option}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("Ninguna opción proporcionada. Usando la opción por defecto 1.")
        return MENU_OPTIONS[default_option]

    if choice == "":
        choice = default_option

    if choice not in MENU_OPTIONS:
        print("Opción inválida. Usando la opción por defecto 1.")
        return MENU_OPTIONS[default_option]

    return MENU_OPTIONS[choice]


def get_strategy_dirs(strategy_key: str) -> tuple[Path, Path, Path]:
    """Devuelve las rutas de las carpetas para tablas, figuras y outputs específicas de la
    estrategia dada."""
    tables_dir = TABLES_DIR / strategy_key
    figures_dir = FIGURES_DIR / strategy_key
    outputs_dir = OUTPUTS_DIR / strategy_key

    for directory in (tables_dir, figures_dir, outputs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return tables_dir, figures_dir, outputs_dir


def _build_distribution_summary(distributions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Construye un DataFrame resumen que consolida las distribuciones de clases por codificación
    en un formato tabular más amigable para su análisis y visualización."""
    rows: list[dict[str, float | int | str]] = []
    for codification, distribution in distributions.items():
        for class_label, payload in distribution.iterrows():
            rows.append(
                {
                    "codification": codification,
                    "class": class_label,
                    "count": int(payload["count"]),
                    "ratio": float(payload["ratio"]),
                }
            )
    return pd.DataFrame(rows)


def save_outputs(
    strategy_key: str,
    strategy_label: str,
    strategy_estimate: str,
    distributions: dict[str, pd.DataFrame],
    feature_selection_outputs: dict[str, pd.DataFrame],
    feature_selection_metadata: dict[str, int],
    pca_explained_df: pd.DataFrame,
    pca_95_df: pd.DataFrame,
    pca_95_components: int,
    codification_metrics_df: pd.DataFrame,
    selected_features_df: pd.DataFrame,
    ensemble_metrics_df: pd.DataFrame,
    confusion_by_model: dict,
    ensemble_cv_splits: int,
    rows: int,
    feature_columns: int,
    elapsed_seconds: float,
) -> dict[str, Path]:
    """Guarda todos los outputs generados durante la ejecución de una estrategia específica en las
    carpetas correspondientes, y devuelve un diccionario con las rutas de los archivos guardados.
    También se encarga de limpiar cualquier artefacto de ejecuciones anteriores que pueda causar
    confusión antes de guardar los nuevos outputs."""
    tables_dir, figures_dir, outputs_dir = get_strategy_dirs(strategy_key)

    legacy_table_files = [
        "model_metrics.csv",
        "confusion_matrix_bagging.csv",
        "confusion_matrix_boosting.csv",
        "confusion_matrix_stacking.csv",
    ]
    legacy_figure_files = [
        "model_metrics.png",
        "confusion_matrix_bagging.png",
        "confusion_matrix_boosting.png",
        "confusion_matrix_stacking.png",
    ]
    for filename in legacy_table_files:
        legacy_path = tables_dir / filename
        if legacy_path.exists():
            legacy_path.unlink()
    for filename in legacy_figure_files:
        legacy_path = figures_dir / filename
        if legacy_path.exists():
            legacy_path.unlink()

    distribution_summary_path = tables_dir / "class_distribution_all_codifications.csv"
    distribution_plot_path = figures_dir / "class_distribution_all_codifications.png"

    codification_metrics_table_path = (
        tables_dir / "naive_bayes_metrics_by_codification.csv"
    )
    codification_metrics_figure_path = (
        figures_dir / "naive_bayes_metrics_by_codification.png"
    )
    selected_features_path = (
        tables_dir / "naive_bayes_selected_features_by_codification.csv"
    )

    ensemble_metrics_table_path = tables_dir / "model_metrics.csv"
    ensemble_metrics_figure_path = figures_dir / "model_metrics.png"

    pca_explained_path = tables_dir / "pca_explained_variance.csv"
    pca_95_path = tables_dir / "pca_components_95.csv"
    pca_plot_path = figures_dir / "pca_cumulative_variance.png"

    feature_variance_path = tables_dir / "features_variance_selected.csv"
    feature_chi2_top15_path = tables_dir / "features_chi2_top15_gds.csv"
    feature_chi2_all_path = tables_dir / "features_chi2_all_gds.csv"
    feature_rfe_logistic_path = tables_dir / "features_rfe_logistic.csv"
    feature_rfe_tree_path = tables_dir / "features_rfe_tree.csv"
    feature_forward_path = tables_dir / "features_forward_knn.csv"
    feature_metadata_path = outputs_dir / "feature_selection_metadata.json"

    summary_path = outputs_dir / "run_summary.json"

    for codification, distribution in distributions.items():
        per_codification_path = (
            tables_dir / f"class_distribution_{codification.lower()}.csv"
        )
        distribution.to_csv(per_codification_path)

    distribution_summary = _build_distribution_summary(distributions)
    distribution_summary.to_csv(distribution_summary_path, index=False)
    plot_grouped_class_distributions(distributions, output_path=distribution_plot_path)

    feature_selection_outputs["variance_selected"].to_csv(
        feature_variance_path, index=False
    )
    feature_selection_outputs["chi2_top15"].to_csv(feature_chi2_top15_path, index=False)
    feature_selection_outputs["chi2_all"].to_csv(feature_chi2_all_path, index=False)
    feature_selection_outputs["rfe_logistic"].to_csv(
        feature_rfe_logistic_path, index=False
    )
    feature_selection_outputs["rfe_tree"].to_csv(feature_rfe_tree_path, index=False)
    feature_selection_outputs["forward_knn"].to_csv(feature_forward_path, index=False)

    feature_metadata_path.write_text(
        json.dumps(feature_selection_metadata, indent=2),
        encoding="utf-8",
    )

    pca_explained_df.to_csv(pca_explained_path, index=False)
    pca_95_df.to_csv(pca_95_path, index=False)
    plot_pca_cumulative_variance(pca_explained_df, output_path=pca_plot_path)

    codification_metrics_df.to_csv(codification_metrics_table_path)
    plot_metrics(
        codification_metrics_df[
            ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        ],
        output_path=codification_metrics_figure_path,
        title="Métricas de Naive Bayes por codificación GDS",
        xlabel="Codificación",
    )
    selected_features_df.to_csv(selected_features_path, index=False)

    ensemble_metrics_df.to_csv(ensemble_metrics_table_path)
    plot_metrics(
        ensemble_metrics_df,
        output_path=ensemble_metrics_figure_path,
        title="Métricas de Ensemble por modelo",
        xlabel="Modelo",
    )

    confusion_table_paths: dict[str, str] = {}
    confusion_figure_paths: dict[str, str] = {}
    for model_name, payload in confusion_by_model.items():
        matrix = payload["matrix"]
        labels = payload["labels"]

        matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
        matrix_table_path = tables_dir / f"confusion_matrix_{model_name}.csv"
        matrix_figure_path = figures_dir / f"confusion_matrix_{model_name}.png"

        matrix_df.to_csv(matrix_table_path)
        plot_confusion_matrix(
            conf_matrix_values=matrix,
            labels=labels,
            title=f"Matriz de Confusión - {model_name.capitalize()}",
            output_path=matrix_figure_path,
        )

        confusion_table_paths[model_name] = relative_path(matrix_table_path)
        confusion_figure_paths[model_name] = relative_path(matrix_figure_path)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy_key,
        "strategy_label": strategy_label,
        "estimated_runtime": strategy_estimate,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "dataset": DATASET_FILENAME,
        "target": TARGET_COLUMN,
        "rows": rows,
        "feature_columns": feature_columns,
        "codifications": [cod for cod in CODIFICATION_COLUMNS if cod in distributions],
        "validation": {
            "codification_model": "NaiveBayes manual",
            "ensemble_models": list(ensemble_metrics_df.index),
        },
        "chi2_k_best": CHI2_K_BEST,
        "pca_components_95": pca_95_components,
        "ensemble_cv_splits": int(ensemble_cv_splits),
        "cv_splits_by_codification": {
            cod: int(splits)
            for cod, splits in codification_metrics_df["cv_splits"].to_dict().items()
        },
        "outputs": {
            "distribution_summary": relative_path(distribution_summary_path),
            "distribution_plot": relative_path(distribution_plot_path),
            "feature_selection": {
                "variance": relative_path(feature_variance_path),
                "chi2_top15": relative_path(feature_chi2_top15_path),
                "chi2_all": relative_path(feature_chi2_all_path),
                "rfe_logistic": relative_path(feature_rfe_logistic_path),
                "rfe_tree": relative_path(feature_rfe_tree_path),
                "forward_knn": relative_path(feature_forward_path),
                "metadata": relative_path(feature_metadata_path),
            },
            "pca": {
                "explained_variance": relative_path(pca_explained_path),
                "components_95": relative_path(pca_95_path),
                "plot": relative_path(pca_plot_path),
            },
            "codification_metrics_table": relative_path(
                codification_metrics_table_path
            ),
            "codification_metrics_figure": relative_path(
                codification_metrics_figure_path
            ),
            "selected_features": relative_path(selected_features_path),
            "ensemble_metrics_table": relative_path(ensemble_metrics_table_path),
            "ensemble_metrics_figure": relative_path(ensemble_metrics_figure_path),
            "confusion_tables": confusion_table_paths,
            "confusion_figures": confusion_figure_paths,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "distribution_summary": distribution_summary_path,
        "codification_metrics_table": codification_metrics_table_path,
        "codification_metrics_figure": codification_metrics_figure_path,
        "selected_features": selected_features_path,
        "ensemble_metrics_table": ensemble_metrics_table_path,
        "ensemble_metrics_figure": ensemble_metrics_figure_path,
        "summary": summary_path,
    }


def main() -> None:
    # Asegurar que las carpetas necesarias existen antes de cualquier operación de
    # lectura/escritura
    ensure_project_dirs()

    # Limpiar artefactos de ejecuciones anteriores que podrían causar confusión
    legacy_root_artifacts = [
        TABLES_DIR / "class_distribution_gds.csv",
        TABLES_DIR / "model_metrics.csv",
        TABLES_DIR / "confusion_matrix_bagging.csv",
        TABLES_DIR / "confusion_matrix_boosting.csv",
        TABLES_DIR / "confusion_matrix_stacking.csv",
        TABLES_DIR / "strategy_model_comparison.csv",
        FIGURES_DIR / "model_metrics.png",
        FIGURES_DIR / "confusion_matrix_bagging.png",
        FIGURES_DIR / "confusion_matrix_boosting.png",
        FIGURES_DIR / "confusion_matrix_stacking.png",
    ]
    for artifact_path in legacy_root_artifacts:
        if artifact_path.exists():
            artifact_path.unlink()

    df = load_spss_dataframe(DATASET_FILENAME)
    x_frame = feature_frame(df)
    x_target, y_target = split_features_target(df, target_col=TARGET_COLUMN)
    validate_binary_features(x_frame)
    distributions = build_codification_distributions(df)

    feature_selection_outputs, feature_selection_metadata = run_feature_selection_study(
        df,
        target_col=TARGET_COLUMN,
        random_state=RANDOM_STATE,
    )
    pca_explained_df, pca_95_df, pca_95_components = run_pca_study(df)

    selected_strategies = choose_execution_strategies()
    run_records: list[dict] = []
    codification_comparison_rows: list[dict[str, float | str | int]] = []
    model_comparison_rows: list[dict[str, float | str | int]] = []

    print(f"Filas: {len(df)}")
    print(f"Columnas de características: {x_frame.shape[1]}")
    print(f"Objetivo base: {TARGET_COLUMN}")
    print(
        f"Codificaciones: {', '.join([cod for cod in CODIFICATION_COLUMNS if cod in df.columns])}"
    )
    print(
        "Métodos de selección de características: varianza, chi2, RFE (logística/arbol), selección forward"
    )
    print("Método de extracción de características: PCA")
    print(
        f"Modelo de validación: Naive Bayes manual (chi2 top-{CHI2_K_BEST} por codificación)"
    )
    print("Conjunto de modelos obligatorio: Bagging, Boosting, Stacking")

    for strategy_key in selected_strategies:
        strategy_label = STRATEGY_DEFINITIONS[strategy_key]["label"]
        strategy_estimate = STRATEGY_DEFINITIONS[strategy_key]["estimate"]

        print("\n" + "=" * 80)
        print(f" Estrategia: {strategy_label}")
        print(f" Tiempo estimado de ejecución: {strategy_estimate}")

        started_at = perf_counter()
        codification_metrics_df, selected_features_df = (
            evaluate_naive_bayes_codifications(
                df=df,
                strategy=strategy_key,
                requested_splits=DESIRED_CV_SPLITS,
                random_state=RANDOM_STATE,
                k_best=CHI2_K_BEST,
            )
        )

        ensemble_metrics_df, confusion_by_model, ensemble_cv_splits = run_model_suite(
            x_frame=x_target,
            y_series=y_target,
            model_builders=model_builders(),
            strategy=strategy_key,
            requested_splits=DESIRED_CV_SPLITS,
            random_state=RANDOM_STATE,
            apply_oversampling=APPLY_OVERSAMPLING,
        )
        elapsed_seconds = perf_counter() - started_at

        output_paths = save_outputs(
            strategy_key=strategy_key,
            strategy_label=strategy_label,
            strategy_estimate=strategy_estimate,
            distributions=distributions,
            feature_selection_outputs=feature_selection_outputs,
            feature_selection_metadata=feature_selection_metadata,
            pca_explained_df=pca_explained_df,
            pca_95_df=pca_95_df,
            pca_95_components=pca_95_components,
            codification_metrics_df=codification_metrics_df,
            selected_features_df=selected_features_df,
            ensemble_metrics_df=ensemble_metrics_df,
            confusion_by_model=confusion_by_model,
            ensemble_cv_splits=ensemble_cv_splits,
            rows=len(df),
            feature_columns=x_frame.shape[1],
            elapsed_seconds=elapsed_seconds,
        )

        for codification, metrics in codification_metrics_df.iterrows():
            codification_comparison_rows.append(
                {
                    "strategy": strategy_key,
                    "codification": codification,
                    "accuracy": float(metrics["accuracy"]),
                    "precision_macro": float(metrics["precision_macro"]),
                    "recall_macro": float(metrics["recall_macro"]),
                    "f1_macro": float(metrics["f1_macro"]),
                    "cv_splits": int(metrics["cv_splits"]),
                    "selected_feature_count": int(metrics["selected_feature_count"]),
                }
            )

        for model_name, metrics in ensemble_metrics_df.iterrows():
            model_comparison_rows.append(
                {
                    "strategy": strategy_key,
                    "model": model_name,
                    "accuracy": float(metrics["accuracy"]),
                    "precision_macro": float(metrics["precision_macro"]),
                    "recall_macro": float(metrics["recall_macro"]),
                    "f1_macro": float(metrics["f1_macro"]),
                    "cv_splits": int(ensemble_cv_splits),
                }
            )

        run_records.append(
            {
                "strategy": strategy_key,
                "label": strategy_label,
                "estimated_runtime": strategy_estimate,
                "elapsed_seconds": round(elapsed_seconds, 2),
                "ensemble_cv_splits": int(ensemble_cv_splits),
                "cv_splits_by_codification": {
                    cod: int(splits)
                    for cod, splits in codification_metrics_df["cv_splits"]
                    .to_dict()
                    .items()
                },
                "outputs": {
                    name: relative_path(path) for name, path in output_paths.items()
                },
            }
        )

        print(f"Completado en {elapsed_seconds:.2f} segundos")
        print(f"Codificaciones evaluadas: {', '.join(codification_metrics_df.index)}")
        print("Métricas de codificación (basadas en macro):")
        print(codification_metrics_df.round(4))
        print("Métricas del conjunto (basadas en macro):")
        print(ensemble_metrics_df.round(4))
        print(f"Divisiones de validación del conjunto: {ensemble_cv_splits}")
        print("Outputs guardados:")
        for key, path in output_paths.items():
            print(f"- {key}: {relative_path(path)}")

    codification_comparison_path = TABLES_DIR / "strategy_codification_comparison.csv"
    codification_comparison_df = pd.DataFrame(codification_comparison_rows)
    codification_comparison_df.to_csv(codification_comparison_path, index=False)

    model_comparison_path = TABLES_DIR / "strategy_model_comparison.csv"
    model_comparison_df = pd.DataFrame(model_comparison_rows)
    model_comparison_df.to_csv(model_comparison_path, index=False)

    global_summary_path = OUTPUTS_DIR / "run_summary.json"
    global_summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_FILENAME,
        "target": TARGET_COLUMN,
        "rows": len(df),
        "feature_columns": x_frame.shape[1],
        "codifications": [cod for cod in CODIFICATION_COLUMNS if cod in df.columns],
        "selected_strategies": selected_strategies,
        "strategy_codification_comparison_table": relative_path(
            codification_comparison_path
        ),
        "strategy_model_comparison_table": relative_path(model_comparison_path),
        "runs": run_records,
    }
    global_summary_path.write_text(
        json.dumps(global_summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print("Pipeline completado")
    print("Outputs guardados:")
    print(
        f"- tabla de comparación de codificaciones: {relative_path(codification_comparison_path)}"
    )
    print(f"- tabla de comparación de modelos: {relative_path(model_comparison_path)}")
    print(f"- resumen global: {relative_path(global_summary_path)}")


if __name__ == "__main__":
    main()
