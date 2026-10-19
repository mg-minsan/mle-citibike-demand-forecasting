from datetime import datetime, timedelta
from pathlib import Path

import pyspark.sql.functions as F

from utils.utils import write_parquet_partition, read_parquet_partition, partition_path
from utils.weather_features import aggregate_daily_weather, aggregate_weekly_weather
from utils.weather_validation import configure_weather_timezone, validate_hourly_weather, validate_weather_dates



def build_daily_weather_features(snapshot_date_str, silver_weather_directory, spark):
    """Read and validate silver hourly weather, then return daily gold aggregates.

    Shared by the standalone weather writer and station/day feature builder;
    this function does not write a separate gold weather partition.
    """
    timezone = configure_weather_timezone(spark)
    df = read_parquet_partition(silver_weather_directory, "silver_weather_", snapshot_date_str, spark)
    # Reject duplicates and incomplete measurements before they distort totals.
    validate_hourly_weather(df, snapshot_date_str, timezone)
    return aggregate_daily_weather(df)


def process_gold_weather_table(snapshot_date_str, silver_weather_directory, gold_weather_directory, spark):
    agg_df = build_daily_weather_features(snapshot_date_str, silver_weather_directory, spark)
    return write_parquet_partition(agg_df, gold_weather_directory, "gold_weather_", snapshot_date_str)



def build_weekly_weather_features(snapshot_date_str, silver_weather_directory, spark):
    """Read daily silver history for [snapshot_date - 7, snapshot_date - 1]."""
    return build_weather_features_for_dates([snapshot_date_str], silver_weather_directory, spark)


def build_weather_features_for_dates(snapshot_date_strs, silver_weather_directory, spark):
    """Batch version for refreshing multiple daily outputs without repeated reads."""
    timezone = configure_weather_timezone(spark)
    snapshots = sorted({datetime.strptime(day, '%Y-%m-%d').date() for day in snapshot_date_strs})
    needed = sorted({day - timedelta(days=lag) for day in snapshots for lag in range(1, 8)})
    paths = [(day, partition_path(silver_weather_directory, 'silver_weather_', day.isoformat(), '.parquet'))
             for day in needed]
    available = [(day, path) for day, path in paths if Path(path).exists()]
    dates = spark.createDataFrame([(day,) for day in snapshots], 'snapshot_date date')
    history = None
    if available:
        hourly = spark.read.parquet(*[path for _, path in available])
        validate_weather_dates(hourly, [day for day, _ in available], timezone)
        history = aggregate_daily_weather(hourly)
    return aggregate_weekly_weather(dates, history)
