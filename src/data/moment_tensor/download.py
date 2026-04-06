from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path

import requests

from src.utils.paths import RAW_MOMENT_TENSOR_DIR


GCMT_MONTHLY_BASE_URL = (
    "https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY"
)
DEFAULT_TIMEOUT = 60


@dataclass(frozen=True)
class GCMTDownloadConfig:
    timeout: int = DEFAULT_TIMEOUT
    base_url: str = GCMT_MONTHLY_BASE_URL


def _gcmt_month_filename(year: int, month: int) -> str:
    month_token = calendar.month_abbr[month].lower()
    year_token = f"{year % 100:02d}"
    return f"{month_token}{year_token}.ndk"


def build_gcmt_monthly_url(
    year: int,
    month: int,
    config: GCMTDownloadConfig | None = None,
) -> str:
    config = config or GCMTDownloadConfig()
    filename = _gcmt_month_filename(year, month)
    return f"{config.base_url}/{year}/{filename}"


def build_gcmt_output_path(
    year: int,
    month: int,
    output_dir: str | Path = RAW_MOMENT_TENSOR_DIR,
) -> Path:
    output_dir = Path(output_dir)
    return output_dir / str(year) / _gcmt_month_filename(year, month)


def download_gcmt_month(
    year: int,
    month: int,
    output_dir: str | Path = RAW_MOMENT_TENSOR_DIR,
    config: GCMTDownloadConfig | None = None,
    refresh: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Download one official Global CMT monthly NDK file.

    This uses the published monthly ASCII catalog files rather than scraping
    the interactive catalog interface.
    """
    config = config or GCMTDownloadConfig()
    output_path = build_gcmt_output_path(year=year, month=month, output_dir=output_dir)

    if output_path.exists() and not refresh:
        if verbose:
            print(f"[SKIP] {output_path}")
        return output_path

    url = build_gcmt_monthly_url(year=year, month=month, config=config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[FETCH] {url}")

    response = requests.get(url, timeout=config.timeout)
    response.raise_for_status()
    output_path.write_text(response.text, encoding="utf-8")

    if verbose:
        print(f"[SAVE] {output_path}")

    return output_path


def download_gcmt_monthly_range(
    start_year: int,
    end_year: int,
    start_month: int = 1,
    end_month: int = 12,
    output_dir: str | Path = RAW_MOMENT_TENSOR_DIR,
    config: GCMTDownloadConfig | None = None,
    refresh: bool = False,
    stop_on_error: bool = False,
    verbose: bool = True,
) -> list[Path]:
    """
    Download Global CMT monthly NDK files over a year-month range.

    For this project's 2015-2025 window, the official monthly catalog is the
    practical primary source.

    TODO:
    - Add optional support for other official catalog bundles if you later
      decide to backfill years outside the monthly archive pattern.
    """
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")

    config = config or GCMTDownloadConfig()
    downloaded: list[Path] = []

    for year in range(start_year, end_year + 1):
        month_from = start_month if year == start_year else 1
        month_to = end_month if year == end_year else 12

        for month in range(month_from, month_to + 1):
            try:
                downloaded.append(
                    download_gcmt_month(
                        year=year,
                        month=month,
                        output_dir=output_dir,
                        config=config,
                        refresh=refresh,
                        verbose=verbose,
                    )
                )
            except requests.HTTPError:
                if stop_on_error:
                    raise
                if verbose:
                    print(f"[WARN] Missing or unavailable GCMT file for {year}-{month:02d}")

    return downloaded


if __name__ == "__main__":
    download_gcmt_monthly_range(
        start_year=2015,
        end_year=2025,
        output_dir=RAW_MOMENT_TENSOR_DIR,
        refresh=False,
        verbose=True,
    )
