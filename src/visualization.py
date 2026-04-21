from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_figure(output_path: str | Path | None) -> None:
    if output_path is None:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)


def plot_metrics(
    results_df: pd.DataFrame,
    output_path: str | Path | None = None,
    title: str = "Comparación de métricas por modelo",
    xlabel: str = "Modelo",
) -> None:
    """Genera un gráfico de barras para comparar las métricas de diferentes modelos."""
    ax = results_df.plot(kind="bar", figsize=(12, 6), colormap="Set2")
    ax.set_title(title)
    ax.set_ylabel("Puntaje")
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    _save_figure(output_path)


def plot_distribution(
    series: pd.Series, title: str, output_path: str | Path | None = None
) -> None:
    ax = series.value_counts().sort_index().plot(kind="bar", figsize=(8, 5))
    ax.set_title(title)
    ax.set_xlabel("Clase")
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    _save_figure(output_path)


def plot_grouped_class_distributions(
    distributions: Mapping[str, pd.DataFrame],
    output_path: str | Path | None = None,
) -> None:
    """Genera un gráfico de barras agrupadas para comparar las distribuciones de clases
    entre diferentes codificaciones."""
    if not distributions:
        return

    class_values = sorted(
        {float(label) for dist in distributions.values() for label in dist.index}
    )
    x_positions = np.arange(len(class_values))
    bar_width = 0.8 / max(1, len(distributions))

    plt.figure(figsize=(10, 6))
    for idx, (codification, dist) in enumerate(distributions.items()):
        counts_map = {
            float(label): int(count) for label, count in dist["count"].to_dict().items()
        }
        counts = [counts_map.get(class_value, 0) for class_value in class_values]
        offset = (idx - (len(distributions) - 1) / 2) * bar_width
        plt.bar(x_positions + offset, counts, width=bar_width, label=codification)

    labels = [
        str(int(value)) if float(value).is_integer() else str(value)
        for value in class_values
    ]
    plt.title("Frecuencias de clases por codificación GDS")
    plt.xlabel("Etiqueta de clase")
    plt.ylabel("Frecuencia")
    plt.xticks(x_positions, labels=labels)
    plt.legend()
    plt.tight_layout()
    _save_figure(output_path)


def plot_pca_cumulative_variance(
    explained_df: pd.DataFrame,
    output_path: str | Path | None = None,
) -> None:
    """Genera un gráfico de línea para mostrar la varianza explicada acumulada por los componentes
    principales."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        explained_df["component"],
        explained_df["cumulative_explained_variance"],
        marker="o",
    )
    plt.title("Varianza explicada acumulada por componentes principales")
    plt.xlabel("Número de componentes principales")
    plt.ylabel("Varianza explicada acumulada")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    _save_figure(output_path)


def plot_confusion_matrix(
    conf_matrix_values: np.ndarray,
    labels: list,
    title: str,
    output_path: str | Path | None = None,
) -> None:
    """Genera un gráfico de matriz de confusión a partir de los valores y etiquetas proporcionados."""
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(conf_matrix_values, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Etiqueta verdadera",
        xlabel="Etiqueta predicha",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = conf_matrix_values.max() / 2.0 if conf_matrix_values.size else 0
    for i in range(conf_matrix_values.shape[0]):
        for j in range(conf_matrix_values.shape[1]):
            value = int(conf_matrix_values[i, j])
            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150)
