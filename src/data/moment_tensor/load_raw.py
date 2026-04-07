from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math

import pandas as pd

from src.utils.io import load_dataframe, save_dataframe
from src.utils.paths import RAW_MOMENT_TENSOR_DIR


def _safe_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_int(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _to_utc_timestamp(date_text: str, time_text: str) -> pd.Timestamp:
    return pd.to_datetime(
        f"{date_text.strip()} {time_text.strip()}",
        utc=True,
        errors="coerce",
    )


def _iter_ndk_blocks(lines: Iterable[str]) -> Iterable[list[str]]:
    buffer: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        buffer.append(line)
        if len(buffer) == 5:
            yield buffer
            buffer = []

    if buffer:
        raise ValueError("NDK file ended with an incomplete 5-line event block.")


def _parse_line1(line: str) -> dict:
    mb_ms_tokens = line[47:55].split()
    mb = _safe_float(mb_ms_tokens[0]) if len(mb_ms_tokens) > 0 else None
    ms = _safe_float(mb_ms_tokens[1]) if len(mb_ms_tokens) > 1 else None

    reference_time = _to_utc_timestamp(line[5:15], line[16:26])

    return {
        "gcmt_reference_catalog": line[0:4].strip(),
        "gcmt_time": reference_time,
        "gcmt_reference_date": line[5:15].strip(),
        "gcmt_reference_time_text": line[16:26].strip(),
        "gcmt_reference_latitude": _safe_float(line[27:33]),
        "gcmt_reference_longitude": _safe_float(line[34:41]),
        "gcmt_reference_depth_km": _safe_float(line[42:47]),
        "gcmt_mb": mb,
        "gcmt_ms": ms,
        "gcmt_location_name": line[56:80].strip(),
    }


def _parse_line2(line: str) -> dict:
    half_duration = None
    moment_rate_type = line[69:80].strip()
    if ":" in moment_rate_type:
        _, duration_text = moment_rate_type.split(":", 1)
        half_duration = _safe_float(duration_text)

    return {
        "gcmt_event_name": line[0:16].strip(),
        "gcmt_data_used": line[17:61].strip(),
        "gcmt_source_type": line[62:68].strip(),
        "gcmt_moment_rate_type": moment_rate_type,
        "gcmt_half_duration_sec": half_duration,
    }


def _parse_line3(line: str, reference_time: pd.Timestamp) -> dict:
    tokens = line.split()
    if len(tokens) < 11 or tokens[0] != "CENTROID:":
        raise ValueError(f"Unexpected NDK centroid line: {line}")

    centroid_time_shift_sec = _safe_float(tokens[1])
    centroid_time = (
        reference_time + pd.to_timedelta(centroid_time_shift_sec, unit="s")
        if pd.notna(reference_time) and centroid_time_shift_sec is not None
        else pd.NaT
    )

    return {
        "gcmt_centroid_time": centroid_time,
        "gcmt_centroid_time_shift_sec": centroid_time_shift_sec,
        "gcmt_centroid_time_shift_error_sec": _safe_float(tokens[2]),
        "gcmt_latitude": _safe_float(tokens[3]),
        "gcmt_latitude_error": _safe_float(tokens[4]),
        "gcmt_longitude": _safe_float(tokens[5]),
        "gcmt_longitude_error": _safe_float(tokens[6]),
        "gcmt_depth_km": _safe_float(tokens[7]),
        "gcmt_depth_error_km": _safe_float(tokens[8]),
        "gcmt_depth_type": tokens[9],
        "gcmt_analysis_timestamp": tokens[10],
    }


def _parse_line4(line: str) -> dict:
    tokens = line.split()
    if len(tokens) < 13:
        raise ValueError(f"Unexpected NDK tensor line: {line}")

    exponent = _safe_int(tokens[0])
    values = [_safe_float(token) for token in tokens[1:13]]

    return {
        "gcmt_moment_exponent": exponent,
        "gcmt_mrr": values[0],
        "gcmt_mrr_error": values[1],
        "gcmt_mtt": values[2],
        "gcmt_mtt_error": values[3],
        "gcmt_mpp": values[4],
        "gcmt_mpp_error": values[5],
        "gcmt_mrt": values[6],
        "gcmt_mrt_error": values[7],
        "gcmt_mrp": values[8],
        "gcmt_mrp_error": values[9],
        "gcmt_mtp": values[10],
        "gcmt_mtp_error": values[11],
    }


def _parse_line5(line: str) -> dict:
    tokens = line.split()
    if len(tokens) < 17:
        raise ValueError(f"Unexpected NDK mechanism line: {line}")

    return {
        "gcmt_version_code": tokens[0],
        "gcmt_eig1": _safe_float(tokens[1]),
        "gcmt_eig1_plunge": _safe_float(tokens[2]),
        "gcmt_eig1_azimuth": _safe_float(tokens[3]),
        "gcmt_eig2": _safe_float(tokens[4]),
        "gcmt_eig2_plunge": _safe_float(tokens[5]),
        "gcmt_eig2_azimuth": _safe_float(tokens[6]),
        "gcmt_eig3": _safe_float(tokens[7]),
        "gcmt_eig3_plunge": _safe_float(tokens[8]),
        "gcmt_eig3_azimuth": _safe_float(tokens[9]),
        "gcmt_scalar_moment": _safe_float(tokens[10]),
        "strike": _safe_float(tokens[11]),
        "dip": _safe_float(tokens[12]),
        "rake": _safe_float(tokens[13]),
        "strike2": _safe_float(tokens[14]),
        "dip2": _safe_float(tokens[15]),
        "rake2": _safe_float(tokens[16]),
    }


def _compute_gcmt_magnitude(scalar_moment: float | None, exponent: int | None) -> float | None:
    if scalar_moment is None or exponent is None or scalar_moment <= 0:
        return None

    moment_dyne_cm = scalar_moment * (10 ** exponent)
    if moment_dyne_cm <= 0:
        return None

    return (2.0 / 3.0) * (math.log10(moment_dyne_cm) - 16.1)


def parse_ndk_record(block: list[str], source_file: str | None = None) -> dict:
    """
    Parse one 5-line Global CMT NDK event block.

    Assumption:
    `gcmt_time` is taken from the reference event time on the first line because
    that aligns best with ComCat origin times for fuzzy matching.
    """
    line1 = _parse_line1(block[0])
    line2 = _parse_line2(block[1])
    line3 = _parse_line3(block[2], reference_time=line1["gcmt_time"])
    line4 = _parse_line4(block[3])
    line5 = _parse_line5(block[4])

    row = {
        **line1,
        **line2,
        **line3,
        **line4,
        **line5,
    }

    row["gcmt_event_name_or_id"] = row["gcmt_event_name"]
    row["gcmt_magnitude"] = _compute_gcmt_magnitude(
        scalar_moment=row["gcmt_scalar_moment"],
        exponent=row["gcmt_moment_exponent"],
    )
    row["gcmt_source_file"] = source_file

    return row


def parse_ndk_file(path: str | Path, verbose: bool = False) -> pd.DataFrame:
    path = Path(path)

    if verbose:
        print(f"[PARSE] {path}")

    with path.open("r", encoding="utf-8") as handle:
        rows = [
            parse_ndk_record(block, source_file=path.name)
            for block in _iter_ndk_blocks(handle)
        ]

    return pd.DataFrame(rows)


def _collect_ndk_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.rglob("*.ndk"))
    if path.suffix.lower() == ".ndk":
        return [path]
    return []


def load_raw_moment_tensor(path: str | Path, verbose: bool = True) -> pd.DataFrame:
    """
    Load raw Global CMT data from an NDK file, an NDK directory, or a saved
    CSV/Parquet extract.
    """
    path = Path(path)

    if path.suffix.lower() in {".csv", ".parquet"}:
        if verbose:
            print(f"[LOAD] Moment tensor file -> {path}")
        return load_dataframe(path)

    ndk_files = _collect_ndk_files(path)
    if not ndk_files:
        raise ValueError(f"No supported moment tensor files found at {path}")

    if verbose:
        if path.is_dir():
            print(f"[LOAD] Moment tensor directory -> {path} ({len(ndk_files)} NDK files)")
        else:
            print(f"[LOAD] Moment tensor file -> {path}")

    frames = [parse_ndk_file(ndk_path, verbose=False) for ndk_path in ndk_files]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("gcmt_time").reset_index(drop=True)

    if verbose:
        print(f"[DONE] Loaded raw moment tensor rows: {len(df):,}")

    return df


def save_raw_moment_tensor_copy(
    input_path: str | Path,
    output_path: str | Path = RAW_MOMENT_TENSOR_DIR / "moment_tensor_raw.parquet",
    verbose: bool = True,
) -> pd.DataFrame:
    df = load_raw_moment_tensor(input_path, verbose=verbose)
    save_dataframe(df, output_path)

    if verbose:
        print(f"[SAVE] {output_path}")

    return df
