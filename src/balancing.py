from __future__ import annotations

import numpy as np


def oversample_minority_classes(
    x_values: np.ndarray,
    y_values: np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Realiza oversampling de las clases minoritarias para balancear el dataset.
    Se utiliza muestreo aleatorio con reemplazo para aumentar el número de muestras
    de las clases minoritarias hasta igualar la cantidad de la clase mayoritaria."""
    labels, counts = np.unique(y_values, return_counts=True)
    max_count = int(counts.max())
    rng = np.random.default_rng(random_state)

    sampled_indices: list[np.ndarray] = []
    for label, count in zip(labels, counts):
        class_indices = np.where(y_values == label)[0]
        if int(count) < max_count:
            extra_indices = rng.choice(
                class_indices, size=max_count - int(count), replace=True
            )
            class_indices = np.concatenate([class_indices, extra_indices])
        sampled_indices.append(class_indices)

    balanced_indices = np.concatenate(sampled_indices)
    rng.shuffle(balanced_indices)

    return x_values[balanced_indices], y_values[balanced_indices]
