from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

from src.data.comcat.clean import save_clean_comcat
from src.data.merge.enrich import merge_comcat_with_moment_tensor
from src.data.moment_tensor.clean import save_clean_moment_tensor
from src.data.moment_tensor.load_raw import load_raw_moment_tensor
from src.dataset.assemble import assemble_modeling_dataset
from src.dataset.features import build_trigger_features
from src.dataset.labels import build_aftershock_labels
from src.dataset.splits import assign_data_split
from src.dataset.triggers import build_trigger_events
from src.utils.io import load_dataframe, save_dataframe
from src.utils.paths import (
    INTERIM_COMCAT_DIR,
    INTERIM_ENRICHED_DIR,
    INTERIM_FEATURES_DIR,
    INTERIM_LABELS_DIR,
    INTERIM_MOMENT_TENSOR_DIR,
    INTERIM_TRIGGERS_DIR,
    PROCESSED_DATASETS_DIR,
    PROCESSED_SPLITS_DIR,
)


DEFAULT_DATASET_NAME = "earthquake_aftershock_v2"


def _slugify_dataset_name(dataset_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", dataset_name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("dataset_name must contain at least one letter or number.")
    return slug


def _require_existing_path(path_like: str | Path | None, label: str) -> Path | None:
    if path_like is None:
        return None

    path = Path(path_like).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")

    return path


def _build_output_paths(dataset_slug: str) -> dict[str, Path]:
    return {
        "clean_comcat": INTERIM_COMCAT_DIR / f"{dataset_slug}__comcat_clean.parquet",
        "clean_comcat_log": INTERIM_COMCAT_DIR / f"{dataset_slug}__comcat_cleaning_log.csv",
        "raw_moment_tensor": INTERIM_MOMENT_TENSOR_DIR / f"{dataset_slug}__moment_tensor_raw.parquet",
        "clean_moment_tensor": INTERIM_MOMENT_TENSOR_DIR / f"{dataset_slug}__moment_tensor_clean.parquet",
        "clean_moment_tensor_log": INTERIM_MOMENT_TENSOR_DIR / f"{dataset_slug}__moment_tensor_cleaning_log.csv",
        "enriched_events": INTERIM_ENRICHED_DIR / f"{dataset_slug}__events_enriched.parquet",
        "triggers": INTERIM_TRIGGERS_DIR / f"{dataset_slug}__triggers.parquet",
        "labels": INTERIM_LABELS_DIR / f"{dataset_slug}__labels.parquet",
        "features": INTERIM_FEATURES_DIR / f"{dataset_slug}__features.parquet",
        "final_dataset": PROCESSED_DATASETS_DIR / f"{dataset_slug}.parquet",
        "split_dir": PROCESSED_SPLITS_DIR / dataset_slug,
    }


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message)


def _load_or_parse_raw_moment_tensor(
    moment_tensor_input_path: Path,
    output_path: Path,
    force_recompute: bool,
    verbose: bool,
) -> pd.DataFrame:
    if output_path.exists() and not force_recompute:
        _log(verbose, f"[LOAD EXISTING] {output_path}")
        return load_dataframe(output_path)

    raw_mt_df = load_raw_moment_tensor(moment_tensor_input_path, verbose=verbose)
    save_dataframe(raw_mt_df, output_path)
    _log(verbose, f"[SAVE] Raw moment tensor copy -> {output_path}")
    return raw_mt_df


def _save_split_exports(
    final_df: pd.DataFrame,
    split_dir: Path,
    dataset_slug: str,
    verbose: bool,
) -> dict[str, Path]:
    split_dir.mkdir(parents=True, exist_ok=True)

    split_paths: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        split_df = final_df.loc[final_df["split"] == split_name].copy()
        split_path = split_dir / f"{dataset_slug}__{split_name}.parquet"
        save_dataframe(split_df, split_path)
        split_paths[split_name] = split_path
        _log(verbose, f"[SAVE] {split_name} split -> {split_path} ({len(split_df):,} rows)")

    return split_paths


def build_v2_dataset(
    comcat_input_path: str | Path,
    moment_tensor_input_path: str | Path | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    train_start_year: int = 2010,
    train_end_year: int = 2022,
    validation_years: tuple[int, ...] = (2023,),
    test_years: tuple[int, ...] = (2024, 2025),
    force_recompute: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the shared preprocessing pipeline end to end and save reusable outputs.

    The pipeline keeps ComCat as the master dataset, optionally enriches it
    with GCMT moment tensor fields, then builds trigger-level labels, features,
    and the final train/val/test-ready modeling table.
    """
    comcat_input_path = _require_existing_path(comcat_input_path, "ComCat input")
    moment_tensor_input_path = _require_existing_path(moment_tensor_input_path, "Moment tensor input")
    dataset_slug = _slugify_dataset_name(dataset_name)
    output_paths = _build_output_paths(dataset_slug)

    _log(verbose, f"[BUILD] dataset_name={dataset_name}")
    _log(verbose, f"[BUILD] dataset_slug={dataset_slug}")
    _log(verbose, f"[BUILD] ComCat input -> {comcat_input_path}")
    if moment_tensor_input_path is None:
        _log(verbose, "[BUILD] GCMT enrichment -> skipped (no moment tensor input provided)")
    else:
        _log(verbose, f"[BUILD] GCMT input -> {moment_tensor_input_path}")

    if output_paths["final_dataset"].exists() and not force_recompute:
        _log(verbose, f"[LOAD EXISTING] {output_paths['final_dataset']}")
        final_df = load_dataframe(output_paths["final_dataset"])
        _save_split_exports(final_df, output_paths["split_dir"], dataset_slug, verbose=verbose)
        return final_df

    comcat_clean_df, _ = save_clean_comcat(
        input_path=comcat_input_path,
        output_path=output_paths["clean_comcat"],
        log_output_path=output_paths["clean_comcat_log"],
        force_recompute=force_recompute,
        verbose=verbose,
    )

    if moment_tensor_input_path is not None:
        raw_mt_df = _load_or_parse_raw_moment_tensor(
            moment_tensor_input_path=moment_tensor_input_path,
            output_path=output_paths["raw_moment_tensor"],
            force_recompute=force_recompute,
            verbose=verbose,
        )
        mt_clean_df, _ = save_clean_moment_tensor(
            raw_mt_df,
            output_path=output_paths["clean_moment_tensor"],
            log_output_path=output_paths["clean_moment_tensor_log"],
            force_recompute=force_recompute,
            verbose=verbose,
        )
    else:
        mt_clean_df = None

    enriched_df = merge_comcat_with_moment_tensor(comcat_clean_df, mt_clean_df)
    save_dataframe(enriched_df, output_paths["enriched_events"])
    _log(verbose, f"[SAVE] Enriched events -> {output_paths['enriched_events']}")

    triggers_df = build_trigger_events(enriched_df)
    save_dataframe(triggers_df, output_paths["triggers"])
    _log(verbose, f"[SAVE] Triggers -> {output_paths['triggers']}")

    labels_df = build_aftershock_labels(
        triggers_df=triggers_df,
        events_df=comcat_clean_df,
        verbose=verbose,
    )
    save_dataframe(labels_df, output_paths["labels"])
    _log(verbose, f"[SAVE] Labels -> {output_paths['labels']}")

    features_df = build_trigger_features(
        triggers_df=triggers_df,
        events_df=comcat_clean_df,
    )
    save_dataframe(features_df, output_paths["features"])
    _log(verbose, f"[SAVE] Features -> {output_paths['features']}")

    final_df = assemble_modeling_dataset(
        triggers_df=triggers_df,
        labels_df=labels_df,
        features_df=features_df,
    )
    final_df = assign_data_split(
        final_df,
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        validation_years=validation_years,
        test_years=test_years,
    )
    save_dataframe(final_df, output_paths["final_dataset"])
    _log(verbose, f"[SAVE] Final dataset -> {output_paths['final_dataset']}")

    _save_split_exports(final_df, output_paths["split_dir"], dataset_slug, verbose=verbose)

    _log(verbose, f"[DONE] Final dataset rows: {len(final_df):,}")
    split_counts = final_df["split"].value_counts(dropna=False).to_dict()
    _log(verbose, f"[DONE] Split counts: {split_counts}")

    return final_df


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the CLI for the shared preprocessing pipeline entrypoint.
    """
    parser = argparse.ArgumentParser(
        description="Build the shared trigger-level earthquake aftershock dataset.",
    )
    parser.add_argument(
        "--comcat-input",
        required=True,
        help="Path to a raw ComCat file or directory (.geojson, .csv, .parquet, or a folder containing them).",
    )
    parser.add_argument(
        "--moment-tensor-input",
        default=None,
        help="Optional GCMT file or directory (.ndk, .csv, .parquet, or a folder containing supported files).",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Human-readable dataset name used to derive saved output filenames.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Rebuild all stages even if cached outputs already exist.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output.",
    )
    parser.add_argument(
        "--train-start-year",
        type=int,
        default=2010,
        help="First training year (inclusive).",
    )
    parser.add_argument(
        "--train-end-year",
        type=int,
        default=2022,
        help="Last training year (inclusive).",
    )
    parser.add_argument(
        "--validation-year",
        type=int,
        default=2023,
        help="Validation year.",
    )
    parser.add_argument(
        "--test-years",
        type=int,
        nargs="+",
        default=[2024, 2025],
        help="One or more test years.",
    )
    return parser


def main() -> None:
    """
    CLI entrypoint for the canonical shared preprocessing pipeline.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    build_v2_dataset(
        comcat_input_path=args.comcat_input,
        moment_tensor_input_path=args.moment_tensor_input,
        dataset_name=args.dataset_name,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        validation_years=(args.validation_year,),
        test_years=tuple(args.test_years),
        force_recompute=args.force_recompute,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
