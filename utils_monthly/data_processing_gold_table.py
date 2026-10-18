"""Monthly gold partitions retaining station/day labels and features."""
from datetime import timedelta
from pathlib import Path

import pyspark.sql.functions as F
from pyspark import StorageLevel

from utils.data_processing_gold_table import ACTIVE_WINDOW_DAYS
from utils.holiday_features import add_holiday_features
from utils_monthly.data_processing_gold_weather_table import build_weekly_weather_features
from utils_monthly.utils import month_bounds, generate_snapshot_dates, monthly_partition_path, write_parquet_partition


def load_daily_trip_starts(snapshot_date_str, silver_trips_directory, spark):
    start, end = month_bounds(snapshot_date_str)
    first = start - timedelta(days=ACTIVE_WINDOW_DAYS)
    paths = [monthly_partition_path(silver_trips_directory, 'silver_trips', month)
             for month in generate_snapshot_dates(first.isoformat(), start.isoformat())]
    paths = [path for path in paths if Path(path).exists()]
    if not paths:
        return spark.createDataFrame([], 'station_id string, start_date date, trip_starts long')
    return (spark.read.parquet(*paths)
            .filter((F.col('start_date') >= F.lit(first)) & (F.col('start_date') < F.lit(end)))
            .groupBy(F.col('start_station_id').alias('station_id'), 'start_date')
            .agg(F.count('*').alias('trip_starts')))


def build_station_days(snapshot_date_str, daily_starts, spark):
    start, end = month_bounds(snapshot_date_str)
    dates = spark.range((end - start).days).select(
        F.date_add(F.lit(start), F.col('id').cast('int')).alias('snapshot_date'))
    # A station joins after its first trip and expires after 28 inactive days.
    return (daily_starts.crossJoin(F.broadcast(dates))
            .filter((F.col('start_date') >= F.date_sub('snapshot_date', ACTIVE_WINDOW_DAYS)) &
                    (F.col('start_date') < F.col('snapshot_date')))
            .select('station_id', 'snapshot_date').distinct())


def build_trip_gold(snapshot_date_str, silver_trips_directory, spark):
    daily = load_daily_trip_starts(snapshot_date_str, silver_trips_directory, spark)
    grid = build_station_days(snapshot_date_str, daily, spark).alias('grid')
    history = daily.alias('history')
    joined = grid.join(history,
                       (F.col('grid.station_id') == F.col('history.station_id')) &
                       (F.col('history.start_date') >= F.date_sub(F.col('grid.snapshot_date'), ACTIVE_WINDOW_DAYS)) &
                       (F.col('history.start_date') <= F.col('grid.snapshot_date')), 'left')
    days_ago = F.datediff(F.col('grid.snapshot_date'), F.col('history.start_date'))

    def total(first, last):
        return F.sum(F.when(days_ago.between(first, last), F.col('history.trip_starts')).otherwise(0))

    return joined.groupBy(F.col('grid.station_id').alias('station_id'),
                          F.col('grid.snapshot_date').alias('snapshot_date')).agg(
        total(0, 0).cast('int').alias('label'),
        total(1, 1).cast('int').alias('fe_lag_1'),
        total(7, 7).cast('int').alias('fe_lag_7'),
        total(14, 14).cast('int').alias('fe_lag_14'),
        (total(1, 7) / 7).cast('double').alias('fe_rolling_mean_7'),
        (total(1, 28) / 28).cast('double').alias('fe_rolling_mean_28'),
    )


def process_labels_gold_table(snapshot_date_str, silver_trips_directory, gold_label_store_directory, spark):
    df = (build_trip_gold(snapshot_date_str, silver_trips_directory, spark)
          .select('station_id', 'label', F.lit('trip_starts_1d').alias('label_def'), 'snapshot_date'))
    return write_parquet_partition(df, gold_label_store_directory, 'gold_label_store', snapshot_date_str)


def process_features_gold_table(snapshot_date_str, silver_trips_directory, gold_feature_store_directory, spark,
                                silver_weather_directory='datamart_monthly/silver/weather'):
    df = build_trip_gold(snapshot_date_str, silver_trips_directory, spark).drop('label')
    df = (df.withColumn('fe_day_of_week', F.dayofweek('snapshot_date'))
          .withColumn('fe_is_weekend', F.col('fe_day_of_week').isin(1, 7).cast('int'))
          .withColumn('fe_month', F.month('snapshot_date')))
    df = add_holiday_features(df, include_long_weekend=True)
    weather = build_weekly_weather_features(snapshot_date_str, silver_weather_directory, spark)
    df = df.join(F.broadcast(weather), 'snapshot_date', 'left')
    df = df.persist(StorageLevel.DISK_ONLY)
    try:
        return write_parquet_partition(df, gold_feature_store_directory, 'gold_feature_store', snapshot_date_str)
    finally:
        df.unpersist()
