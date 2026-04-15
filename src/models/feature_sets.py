from __future__ import annotations

"""
Explicit starter feature sets for the modeling stage.

These lists are intentionally conservative and are based on columns produced by
the current shared preprocessing pipeline. Optional GCMT columns are kept in a
separate list because they may be absent in non-enriched datasets or heavily
sparse even when present.
"""

NON_FEATURE_COLUMNS = [
    "trigger_event_id",
    "trigger_time",
    "updated",
    "split",
    "trigger_magnitude_bin",
    "gcmt_match_rule",
    "gcmt_event_name",
    "gcmt_time",
    "gcmt_reference_catalog",
    "gcmt_reference_date",
    "gcmt_reference_time_text",
    "gcmt_location_name",
    "gcmt_data_used",
    "gcmt_source_type",
    "gcmt_moment_rate_type",
    "gcmt_depth_type",
    "gcmt_analysis_timestamp",
    "gcmt_version_code",
    "gcmt_event_name_or_id",
    "gcmt_source_file",
    "magnitude_type",
    "status",
    "event_type",
    "net",
    "y_24h",
    "y_72h",
    "n_aftershocks_24h",
    "n_aftershocks_72h",
    "max_aftershock_magnitude_24h",
    "max_aftershock_magnitude_72h",
]

BASE_TABULAR_FEATURES = [
    "trigger_latitude",
    "trigger_longitude",
    "trigger_depth_km",
    "trigger_magnitude",
    "trigger_month",
    "trigger_dayofyear",
    "trigger_hour",
    "prior_global_event_count_24h",
    "prior_global_event_count_7d",
]

QUALITY_FEATURES = [
    "gap",
    "dmin",
    "rms",
    "nst",
]

# First neural baseline feature set for modeling-stage-only experiments.
NN_CORE_V1 = (
    BASE_TABULAR_FEATURES
    + QUALITY_FEATURES
    + [
        "trigger_year_feature",
    ]
)

GCMT_FEATURES = [
    "has_gcmt",
    "gcmt_time_diff_sec",
    "gcmt_distance_km",
    "gcmt_mag_diff",
    "gcmt_latitude",
    "gcmt_longitude",
    "gcmt_depth_km",
    "gcmt_magnitude",
    "strike",
    "dip",
    "rake",
    "gcmt_half_duration_sec",
    "gcmt_centroid_time_shift_sec",
    "gcmt_centroid_time_shift_error_sec",
    "gcmt_latitude_error",
    "gcmt_longitude_error",
    "gcmt_depth_error_km",
    "gcmt_moment_exponent",
    "gcmt_mrr",
    "gcmt_mrr_error",
    "gcmt_mtt",
    "gcmt_mtt_error",
    "gcmt_mpp",
    "gcmt_mpp_error",
    "gcmt_mrt",
    "gcmt_mrt_error",
    "gcmt_mrp",
    "gcmt_mrp_error",
    "gcmt_mtp",
    "gcmt_mtp_error",
    "gcmt_eig1",
    "gcmt_eig1_plunge",
    "gcmt_eig1_azimuth",
    "gcmt_eig2",
    "gcmt_eig2_plunge",
    "gcmt_eig2_azimuth",
    "gcmt_eig3",
    "gcmt_eig3_plunge",
    "gcmt_eig3_azimuth",
    "gcmt_scalar_moment",
    "strike2",
    "dip2",
    "rake2",
]

# Modeling-stage-only neural feature set with optional GCMT enrichment.
NN_ENRICHED_V1 = NN_CORE_V1 + GCMT_FEATURES

EXTENDED_TABULAR_FEATURES = (
    BASE_TABULAR_FEATURES
    + QUALITY_FEATURES
    + [
        "trigger_year_feature",
    ]
    + GCMT_FEATURES
)

# Smaller, more conservative feature set for the max-magnitude task.
# The goal is to keep the strongest trigger/context variables while avoiding
# the very high-dimensional GCMT tensor/eigenvalue block that can invite noise.
MAXMAG_COMPACT_FEATURES = [
    "trigger_latitude",
    "trigger_longitude",
    "trigger_depth_km",
    "trigger_magnitude",
    "trigger_month",
    "trigger_dayofyear",
    "trigger_hour",
    "prior_global_event_count_24h",
    "prior_global_event_count_7d",
    "gap",
    "dmin",
    "rms",
    "nst",
    "has_gcmt",
    "gcmt_time_diff_sec",
    "gcmt_distance_km",
    "gcmt_mag_diff",
    "gcmt_depth_km",
    "strike",
    "dip",
    "rake",
]
