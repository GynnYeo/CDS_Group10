"""Download raw ComCat and GCMT data if they are missing.

This script is intended as the first-time setup step before running the shared
dataset build pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.comcat.retrieve import ComCatRequestConfig, download_comcat_range
from src.data.moment_tensor.download import download_gcmt_monthly_range
from src.utils.paths import RAW_COMCAT_DIR, RAW_MOMENT_TENSOR_DIR


START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
START_YEAR = 2015
END_YEAR = 2025


def _has_files(root: Path, patterns: tuple[str, ...]) -> bool:
    if not root.exists():
        return False
    return any(any(root.rglob(pattern)) for pattern in patterns)


def main() -> None:
    print("[CHECK] Raw data setup")

    comcat_exists = _has_files(RAW_COMCAT_DIR, ("*.geojson", "*.csv", "*.parquet"))
    gcmt_exists = _has_files(RAW_MOMENT_TENSOR_DIR, ("*.ndk", "*.csv", "*.parquet"))

    if comcat_exists:
        print(f"[SKIP] ComCat raw data already present in {RAW_COMCAT_DIR}")
    else:
        print(f"[DOWNLOAD] ComCat raw data missing -> downloading into {RAW_COMCAT_DIR}")
        comcat_files = download_comcat_range(
            start_date=START_DATE,
            end_date=END_DATE,
            output_dir=RAW_COMCAT_DIR,
            config=ComCatRequestConfig(min_magnitude=2.5),
            refresh=False,
            verbose=True,
        )
        print(f"[DONE] ComCat files available: {len(comcat_files)}")

    if gcmt_exists:
        print(f"[SKIP] GCMT raw data already present in {RAW_MOMENT_TENSOR_DIR}")
    else:
        print(f"[DOWNLOAD] GCMT raw data missing -> downloading into {RAW_MOMENT_TENSOR_DIR}")
        gcmt_files = download_gcmt_monthly_range(
            start_year=START_YEAR,
            end_year=END_YEAR,
            output_dir=RAW_MOMENT_TENSOR_DIR,
            refresh=False,
            verbose=True,
        )
        print(f"[DONE] GCMT files available: {len(gcmt_files)}")

    print("[READY] Raw data check complete")
    print(f"ComCat directory: {RAW_COMCAT_DIR}")
    print(f"GCMT directory: {RAW_MOMENT_TENSOR_DIR}")


if __name__ == "__main__":
    main()