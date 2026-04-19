from pathlib import Path

import pandas as pd

from .config import dataset_path


def load_spss_dataframe(file_path: str | Path | None = None) -> pd.DataFrame:
    if file_path is None:
        path = dataset_path()
    else:
        provided = Path(file_path)
        path = provided if provided.exists() else dataset_path(provided.name)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_spss(path)
