"""Download raw ComCat and GCMT data if they are missing."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.comcat.retrieve import ComCatRequestConfig, download_comcat_range
from src.data.moment_tensor.download import download_gcmt_monthly_range
from src.utils.paths import RAW_COMCAT_DIR, RAW_MOMENT_TENSOR_DIR


START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
START_YEAR = 2010
END_YEAR = 2025


def main() -> None:
    print("[CHECK] Raw data setup")

    print(f"[DOWNLOAD/CHECK] ComCat -> {RAW_COMCAT_DIR}")
    comcat_files = download_comcat_range(
        start_date=START_DATE,
        end_date=END_DATE,
        output_dir=RAW_COMCAT_DIR,
        config=ComCatRequestConfig(min_magnitude=2.5),
        refresh=False,
        verbose=True,
    )
    print(f"[DONE] ComCat files available/checked: {len(comcat_files)}")

    print(f"[DOWNLOAD/CHECK] GCMT -> {RAW_MOMENT_TENSOR_DIR}")
    gcmt_files = download_gcmt_monthly_range(
        start_year=START_YEAR,
        end_year=END_YEAR,
        output_dir=RAW_MOMENT_TENSOR_DIR,
        refresh=False,
        verbose=True,
    )
    print(f"[DONE] GCMT files available/checked: {len(gcmt_files)}")

    print("[READY] Raw data check complete")
    print(f"ComCat directory: {RAW_COMCAT_DIR}")
    print(f"GCMT directory: {RAW_MOMENT_TENSOR_DIR}")


if __name__ == "__main__":
    main()