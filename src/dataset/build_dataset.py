from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.data.comcat.clean import save_clean_comcat
from src.data.moment_tensor.clean import save_clean_moment_tensor
from src.data.moment_tensor.load_raw import load_raw_moment_tensor
from src.data.merge.enrich import merge_comcat_with_moment_tensor
from src.dataset.triggers import build_trigger_events
from src.dataset.labels import build_aftershock_labels
from src.dataset.features import build_trigger_features
from src.dataset.assemble import assemble_modeling_dataset
from src.dataset.splits import assign_data_split
from src.utils.io import save_dataframe
from src.utils.paths import (
    RAW_COMCAT_DIR,
    INTERIM_COMCAT_DIR,
    INTERIM_MOMENT_TENSOR_DIR,
    INTERIM_ENRICHED_DIR,
    INTERIM_TRIGGERS_DIR,
    INTERIM_LABELS_DIR,
    INTERIM_FEATURES_DIR,
    PROCESSED_DATASETS_DIR,
    PROCESSED_SPLITS_DIR,
)


def build_v2_dataset(
    comcat_input_path: str | Path,
    moment_tensor_input_path: str | Path | None = None,
    dataset_name: str = "earthquake_aftershock_v2",
    force_recompute: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Orchestrate the canonical v2 dataset pipeline.

    Outputs are saved explicitly at each stage so the project remains easy to
    debug and present in a school setting.
    """
    dataset_slug = dataset_name.strip().replace(" ", "_")

    clean_comcat_path = INTERIM_COMCAT_DIR / f"{dataset_slug}__comcat_clean.parquet"
    clean_comcat_log_path = INTERIM_COMCAT_DIR / f"{dataset_slug}__comcat_cleaning_log.csv"
    enriched_path = INTERIM_ENRICHED_DIR / f"{dataset_slug}__events_enriched.parquet"
    triggers_path = INTERIM_TRIGGERS_DIR / f"{dataset_slug}__triggers.parquet"
    labels_path = INTERIM_LABELS_DIR / f"{dataset_slug}__labels.parquet"
    features_path = INTERIM_FEATURES_DIR / f"{dataset_slug}__features.parquet"
    final_dataset_path = PROCESSED_DATASETS_DIR / f"{dataset_slug}.parquet"
    split_dir = PROCESSED_SPLITS_DIR / dataset_slug

    if final_dataset_path.exists() and not force_recompute:
        if verbose:
            print(f"[LOAD EXISTING] {final_dataset_path}")
        return pd.read_parquet(final_dataset_path)

    comcat_clean_df, _ = save_clean_comcat(
        input_path=comcat_input_path,
        output_path=clean_comcat_path,
        log_output_path=clean_comcat_log_path,
        force_recompute=force_recompute,
        verbose=verbose,
    )

    if moment_tensor_input_path is not None:
        raw_mt_df = load_raw_moment_tensor(moment_tensor_input_path, verbose=verbose)
        save_dataframe(
            raw_mt_df,
            INTERIM_MOMENT_TENSOR_DIR / f"{dataset_slug}__moment_tensor_raw.parquet",
        )
        mt_clean_df, _ = save_clean_moment_tensor(
            raw_mt_df,
            output_path=INTERIM_MOMENT_TENSOR_DIR / f"{dataset_slug}__moment_tensor_clean.parquet",
            log_output_path=INTERIM_MOMENT_TENSOR_DIR / f"{dataset_slug}__moment_tensor_cleaning_log.csv",
            verbose=verbose,
        )
    else:
        mt_clean_df = None

    enriched_df = merge_comcat_with_moment_tensor(comcat_clean_df, mt_clean_df)
    save_dataframe(enriched_df, enriched_path)

    triggers_df = build_trigger_events(enriched_df)
    save_dataframe(triggers_df, triggers_path)

    labels_df = build_aftershock_labels(triggers_df=triggers_df, events_df=comcat_clean_df, verbose=verbose)
    save_dataframe(labels_df, labels_path)

    features_df = build_trigger_features(triggers_df, events_df=comcat_clean_df)
    save_dataframe(features_df, features_path)

    final_df = assemble_modeling_dataset(
        triggers_df=triggers_df,
        labels_df=labels_df,
        features_df=features_df,
    )
    final_df = assign_data_split(final_df)
    save_dataframe(final_df, final_dataset_path)
    split_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        split_df = final_df.loc[final_df["split"] == split_name].copy()
        save_dataframe(split_df, split_dir / f"{split_name}.parquet")

    if verbose:
        print(f"[DONE] Final dataset rows: {len(final_df):,}")
        print(f"[DONE] Final dataset path: {final_dataset_path}")

    return final_df


if __name__ == "__main__":
    # TODO:
    # - Point this to the canonical combined ComCat parquet for 2015-2025.
    # - Optionally provide a moment tensor extract once its source schema is fixed.
    default_comcat_input = RAW_COMCAT_DIR / "comcat_2015_2025.parquet"

    if default_comcat_input.exists():
        build_v2_dataset(
            comcat_input_path=default_comcat_input,
            moment_tensor_input_path=None,
            force_recompute=False,
        )
    else:
        print(f"[TODO] Set a real ComCat input file before running: {default_comcat_input}")
