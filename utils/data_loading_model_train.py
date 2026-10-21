import os
from datetime import datetime, timedelta

import pyspark.sql.functions as F

# PURPOSE
# One place that builds the model dataset (features + label) for a date range, so that training, validation,
# testing and the daily prediction job all read data the same way and use exactly the same feature list.
#
# WHAT IT DOES
# 1. reads the dated gold feature partitions   gold_feature_store_YYYY_MM_DD.parquet  for every date in the range
# 2. reads the dated gold label partitions     gold_label_store_YYYY_MM_DD.parquet    for every date in the range
# 3. joins features and labels on (snapshot_date, station_id)
# 4. runs data checks and fails loudly if anything is off (missing partition, row count change, nulls, duplicates)
#
# HOW TO USE (from inside scripts/):
#   df = load_model_dataset("2016-01-30", "2016-05-31", feature_dir, label_dir, spark)
# returns a Spark dataframe with: snapshot_date, station_id, label, then the 22 model inputs in FEATURE_COLUMNS.
# For the daily prediction job there is no label yet: use load_feature_dataset (same features, no label join).

KEY_COLUMNS = ["snapshot_date", "station_id"]
LABEL_COLUMN = "label"

# the 22 model inputs, in a fixed order, so training and inference always see the same columns
FEATURE_COLUMNS = [
    # trip history (per station)
    "fe_lag_1", "fe_lag_7", "fe_lag_14", "fe_rolling_mean_7", "fe_rolling_mean_28",
    # calendar
    "fe_day_of_week", "fe_is_weekend", "fe_month",
    "fe_is_holiday", "fe_is_observed_holiday", "fe_is_long_weekend",
    # weather (daily, citywide)
    "mean_temperature_c", "med_temperature_c", "max_temperature_c", "min_temperature_c", "temperature_range_c",
    "total_precipitation_mm", "total_rain_mm", "mean_cloud_pct",
    "mean_wind_speed_kmh", "max_wind_speed_kmh", "wet_hours",
]


def list_partition_paths(directory, prefix, start_date_str, end_date_str):
    # one dated partition folder per day in the inclusive range; fail loudly if any day is missing
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    if start_date > end_date:
        raise ValueError("start date must be on or before end date")

    paths = []
    missing = []
    for offset in range((end_date - start_date).days + 1):
        day = start_date + timedelta(days=offset)
        path = os.path.join(directory, prefix + day.strftime("%Y_%m_%d") + ".parquet")
        if os.path.isdir(path):
            paths.append(path)
        else:
            missing.append(path)

    if len(missing) > 0:
        raise FileNotFoundError(str(len(missing)) + " partitions missing, first: " + missing[0])
    return paths


def load_feature_dataset(start_date_str, end_date_str, feature_dir, spark):
    # features (incl. holiday flags) for the date range, no label; also used by the daily prediction job
    paths = list_partition_paths(feature_dir, "gold_feature_store_", start_date_str, end_date_str)
    df = spark.read.parquet(*paths)

    missing_columns = [c for c in KEY_COLUMNS + FEATURE_COLUMNS if c not in df.columns]
    if len(missing_columns) > 0:
        raise ValueError("feature columns missing: " + str(missing_columns))
    return df.select(*KEY_COLUMNS, *FEATURE_COLUMNS)


def load_model_dataset(start_date_str, end_date_str, feature_dir, label_dir, spark):
    # features + label for the date range (training, validation, testing)
    features_df = load_feature_dataset(start_date_str, end_date_str, feature_dir, spark)
    label_paths = list_partition_paths(label_dir, "gold_label_store_", start_date_str, end_date_str)
    labels_df = spark.read.parquet(*label_paths).select(*KEY_COLUMNS, LABEL_COLUMN)

    df = features_df.join(labels_df, on=KEY_COLUMNS, how="inner")

    # data checks: features and labels line up 1:1, no duplicate keys, no nulls, no negative labels
    n_features = features_df.count()
    n_labels = labels_df.count()
    n_rows = df.count()
    n_keys = df.select(*KEY_COLUMNS).distinct().count()
    n_dates = df.select("snapshot_date").distinct().count()
    n_days = (datetime.strptime(end_date_str, "%Y-%m-%d") - datetime.strptime(start_date_str, "%Y-%m-%d")).days + 1
    n_null = df.filter(F.greatest(*[F.col(c).isNull().cast("int") for c in FEATURE_COLUMNS + [LABEL_COLUMN]]) == 1).count()
    n_negative = df.filter(F.col(LABEL_COLUMN) < 0).count()
    print(start_date_str, "to", end_date_str, "model dataset | feature rows:", n_features, "| label rows:", n_labels,
          "| joined rows:", n_rows, "| dates:", n_dates, "of", n_days, "| rows with nulls:", n_null)

    if not (n_features == n_labels == n_rows):
        raise ValueError("features and labels do not line up 1:1")
    if n_keys != n_rows:
        raise ValueError("duplicate (snapshot_date, station_id) rows")
    if n_dates != n_days:
        raise ValueError("some dates in the range have no rows")
    if n_null > 0:
        raise ValueError(str(n_null) + " rows have null features or label")
    if n_negative > 0:
        raise ValueError(str(n_negative) + " rows have a negative label")

    return df.select(*KEY_COLUMNS, LABEL_COLUMN, *FEATURE_COLUMNS)