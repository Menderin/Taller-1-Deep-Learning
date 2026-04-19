from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figuras"
TABLES_DIR = REPORTS_DIR / "tablas"

DEFAULT_DATASET_FILENAME = "15 atributos R0-R5.sav"


def dataset_path(filename: str = DEFAULT_DATASET_FILENAME) -> Path:
    candidate_in_raw = RAW_DATA_DIR / filename
    if candidate_in_raw.exists():
        return candidate_in_raw
    return DATA_DIR / filename


def ensure_project_dirs() -> None:
    for folder in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)
