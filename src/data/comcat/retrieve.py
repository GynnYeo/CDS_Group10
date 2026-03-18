from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Iterator
from src.utils.paths import COMCAT_RAW_DIR

import requests


BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
DEFAULT_TIMEOUT = 60
DEFAULT_LIMIT = 20_000



@dataclass(frozen=True)
class ComCatRequestConfig:
    min_magnitude: float = 2.5
    event_type: str = "earthquake"
    order_by: str = "time-asc"
    limit: int = DEFAULT_LIMIT
    timeout: int = DEFAULT_TIMEOUT


def fetch_comcat_chunk(
    starttime: str,
    endtime: str,
    config: ComCatRequestConfig | None = None,
    offset: int = 1,
    session: requests.Session | None = None,
) -> dict:
    """
    Fetch one chunk of ComCat event data as GeoJSON.

    Parameters
    ----------
    starttime : str
        Start date/time in ISO-like string format, e.g. '2025-01-01'.
    endtime : str
        End date/time in ISO-like string format, e.g. '2025-01-31'.
    config : ComCatRequestConfig | None
        Request configuration. Uses defaults if None.
    offset : int
        ComCat pagination offset. Starts at 1.
    session : requests.Session | None
        Optional persistent session for repeated requests.

    Returns
    -------
    dict
        Parsed GeoJSON response.
    """
    config = config or ComCatRequestConfig()
    http = session or requests

    params = {
        "format": "geojson",
        "eventtype": config.event_type,
        "starttime": starttime,
        "endtime": endtime,
        "minmagnitude": config.min_magnitude,
        "orderby": config.order_by,
        "limit": config.limit,
        "offset": offset,
    }

    response = http.get(BASE_URL, params=params, timeout=config.timeout)
    response.raise_for_status()
    return response.json()


def save_raw_geojson(data: dict, filepath: Path) -> None:
    """
    Save GeoJSON response to disk.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def parse_date(value: str) -> date:
    """
    Parse a YYYY-MM-DD string into a date object.
    """
    return datetime.strptime(value, "%Y-%m-%d").date()


def month_ranges(start_date: str, end_date: str) -> Iterator[tuple[str, str, int, int]]:
    """
    Yield monthly date ranges covering [start_date, end_date].

    Returns
    -------
    Iterator[tuple[str, str, int, int]]
        Tuples of:
        (chunk_start_str, chunk_end_str, year, month)
    """
    start = parse_date(start_date)
    end = parse_date(end_date)

    if start > end:
        raise ValueError("start_date must be <= end_date")

    current = start.replace(day=1)

    while current <= end:
        year = current.year
        month = current.month
        last_day = calendar.monthrange(year, month)[1]

        chunk_start = max(start, current)
        chunk_end = min(end, current.replace(day=last_day))

        yield chunk_start.isoformat(), chunk_end.isoformat(), year, month

        if month == 12:
            current = current.replace(year=year + 1, month=1, day=1)
        else:
            current = current.replace(month=month + 1, day=1)


def build_output_filepath(output_dir: Path, year: int, month: int) -> Path:
    """
    Build the output filepath for a monthly GeoJSON file.
    """
    return output_dir / str(year) / f"comcat_{year}_{month:02d}.geojson"


def download_comcat_range(
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    config: ComCatRequestConfig | None = None,
    refresh: bool = False,
    verbose: bool = True,
) -> list[Path]:
    """
    Download ComCat data month by month over a date range.

    Parameters
    ----------
    start_date : str
        Inclusive start date in YYYY-MM-DD format.
    end_date : str
        Inclusive end date in YYYY-MM-DD format.
    output_dir : str | Path
        Root directory where yearly subfolders will be created.
    config : ComCatRequestConfig | None
        Request settings.
    refresh : bool
        If True, redownload files even if they already exist.
    verbose : bool
        If True, print progress messages.

    Returns
    -------
    list[Path]
        Paths of all expected monthly output files.
    """
    config = config or ComCatRequestConfig()
    output_dir = Path(output_dir)

    saved_files: list[Path] = []

    with requests.Session() as session:
        for chunk_start, chunk_end, year, month in month_ranges(start_date, end_date):
            filepath = build_output_filepath(output_dir, year, month)
            saved_files.append(filepath)

            if filepath.exists() and not refresh:
                if verbose:
                    print(f"[SKIP] {filepath}")
                continue

            if verbose:
                print(f"[FETCH] {chunk_start} to {chunk_end} -> {filepath}")

            data = fetch_comcat_chunk(
                starttime=chunk_start,
                endtime=chunk_end,
                config=config,
                session=session,
            )
            save_raw_geojson(data, filepath)

            if verbose:
                n_events = len(data.get("features", []))
                print(f"[DONE] Saved {n_events} events")

    return saved_files


if __name__ == "__main__":
    config = ComCatRequestConfig(min_magnitude=2.5)

    download_comcat_range(
        start_date="2015-01-01",
        end_date="2025-12-31",
        output_dir=COMCAT_RAW_DIR,
        config=config,
        refresh=False,
        verbose=True,
    )