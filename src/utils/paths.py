from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

COMCAT_RAW_DIR = RAW_DIR / "comcat"
COMCAT_INTERIM_DIR = INTERIM_DIR / "comcat"
COMCAT_PROCESSED_DIR = PROCESSED_DIR / "comcat"