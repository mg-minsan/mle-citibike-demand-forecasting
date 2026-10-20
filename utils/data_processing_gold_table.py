import os
import logging
from datetime import datetime, timedelta

import pyspark
from pyspark import StorageLevel
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, DoubleType, DateType

from utils.data_processing_gold_weather_table import build_daily_weather_features
from utils.weather_features import WEATHER_FEATURE_COLUMNS
from utils.holiday_features import add_holiday_features


# how many past days a station must have a trip in to be part of the station x day grid
# (a station with no trips for 28 days is treated as closed; a brand-new station joins after its first trip)
ACTIVE_WINDOW_DAYS = 28

def join_weather_features(features, snapshot_date_str, silver_weather_directory, spark):
    """Join same-day weather to stations without removing station rows."""
    weather = build_daily_weather_features(snapshot_date_str, silver_weather_directory, spark)
    return features.join(F.broadcast(weather), on="snapshot_date", how="left")


def load_daily_trip_starts(snapshot_date, days_back, silver_trips_directory, spark):
    # read the silver files for [snapshot_date - days_back, snapshot_date] and count trips per start station per day
    filepaths = []
    for i in range(days_back, -1, -1):
        day = snapshot_date - timedelta(days=i)
        filepath = silver_trips_directory + "silver_trips_" + day.strftime("%Y_%m_%d") + ".parquet"
        if os.path.exists(filepath):
            filepaths.append(filepath)
    print('silver files found:', len(filepaths), 'of', days_back + 1)

    if len(filepaths) == 0:
        # e.g. the very first day of the history: no earlier silver files yet
        return spark.createDataFrame([], "station_id string, start_date date, trip_starts long")

    df = spark.read.parquet(*filepaths)
    df = df.groupBy(col("start_station_id").alias("station_id"), col("start_date")) \
           .agg(F.count("*").alias("trip_starts"))
    return df


def get_active_stations(daily_starts, snapshot_date):
    # stations with at least one trip in the 28 days before the snapshot date (uses only data up to D-1)
    window_start = (snapshot_date - timedelta(days=ACTIVE_WINDOW_DAYS)).date()
    window_end = (snapshot_date - timedelta(days=1)).date()
    return daily_starts.filter(col("start_date").between(window_start, window_end)) \
                       .select("station_id").distinct()


def process_labels_gold_table(snapshot_date_str, silver_trips_directory, gold_label_store_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to silver table: the snapshot day plus the 28 days before it (to know which stations are active)
    daily_starts = load_daily_trip_starts(snapshot_date, ACTIVE_WINDOW_DAYS, silver_trips_directory, spark)

    # get label: trip starts on the snapshot date for every active station, 0 if it had no trips that day
    stations = get_active_stations(daily_starts, snapshot_date)
    today = daily_starts.filter(col("start_date") == F.lit(snapshot_date_str).cast(DateType()))
    df = stations.join(today.select("station_id", "trip_starts"), on="station_id", how="left")
    df = df.withColumn("label", F.coalesce(col("trip_starts"), F.lit(0)).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit("trip_starts_1d").cast(StringType()))
    df = df.withColumn("snapshot_date", F.lit(snapshot_date_str).cast(DateType()))

    # select columns to save
    df = df.select("station_id", "label", "label_def", "snapshot_date")

    # data checks: stations, zero-filled rows, total trips, no negative labels
    n_rows = df.count()
    n_zero = df.filter(col("label") == 0).count()
    total = df.agg(F.sum("label")).collect()[0][0]
    n_negative = df.filter(col("label") < 0).count()
    print(snapshot_date_str, 'label store | stations:', n_rows, '| zero-trip stations:', n_zero,
          '| total trip_starts:', total, '| negative labels:', n_negative)

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_features_gold_table(snapshot_date_str, silver_trips_directory, gold_feature_store_directory, spark,
                                silver_weather_directory="datamart/silver/weather"):
    """Build station/day trip, calendar and weather features and write gold.

    Weather is aggregated directly from silver using the shared gold weather
    function; no precomputed gold weather partitions are required.
    """
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Trip history uses only the 28 days BEFORE the snapshot date (through D-1).
    daily_starts = load_daily_trip_starts(snapshot_date - timedelta(days=1), ACTIVE_WINDOW_DAYS - 1, silver_trips_directory, spark)
    stations = get_active_stations(daily_starts, snapshot_date)

    def day(n):
        # the date n days before the snapshot date
        return F.lit((snapshot_date - timedelta(days=n)).strftime("%Y-%m-%d")).cast(DateType())

    def starts_between(first_n, last_n):
        # total trip starts from n=first_n to n=last_n days before the snapshot date (missing days count as 0)
        return F.sum(F.when(col("start_date").between(day(last_n), day(first_n)), col("trip_starts")).otherwise(0))

    # feature engineering: lags and rolling averages of daily trip starts
    df = stations.join(daily_starts, on="station_id", how="left").groupBy("station_id").agg(
        starts_between(1, 1).alias("fe_lag_1"),
        starts_between(7, 7).alias("fe_lag_7"),
        starts_between(14, 14).alias("fe_lag_14"),
        (starts_between(1, 7) / 7).alias("fe_rolling_mean_7"),
        (starts_between(1, 28) / 28).alias("fe_rolling_mean_28"),
    )

    # feature engineering: calendar features of the snapshot date
    df = df.withColumn("snapshot_date", F.lit(snapshot_date_str).cast(DateType()))
    df = df.withColumn("fe_day_of_week", F.dayofweek(col("snapshot_date")))  # 1 = Sunday ... 7 = Saturday
    df = df.withColumn("fe_is_weekend", F.col("fe_day_of_week").isin(1, 7).cast(IntegerType()))
    df = df.withColumn("fe_month", F.month(col("snapshot_date")))
    df = add_holiday_features(df, include_long_weekend=True)

    # enforce data types
    for c in ["fe_lag_1", "fe_lag_7", "fe_lag_14"]:
        df = df.withColumn(c, col(c).cast(IntegerType()))
    for c in ["fe_rolling_mean_7", "fe_rolling_mean_28"]:
        df = df.withColumn(c, col(c).cast(DoubleType()))

    df = join_weather_features(df, snapshot_date_str, silver_weather_directory, spark)

    # data checks: stations and nulls
    any_null = F.lit(False)
    for c in df.columns:
        if c not in ("station_id", "snapshot_date"):
            any_null = any_null | col(c).isNull()
    # Reuse the computed result for diagnostics and writing without retaining
    # the full trip history in JVM memory across repeated Spark actions.
    df = df.persist(StorageLevel.DISK_ONLY)
    try:
        stats = df.agg(F.count("*").alias("stations"),
                       F.count(F.when(any_null, 1)).alias("null_rows")).first()
        print(snapshot_date_str, 'feature store | stations:', stats.stations,
              '| rows with null features:', stats.null_rows)

        partition_name = "gold_feature_store_" + snapshot_date_str.replace('-', '_') + '.parquet'
        filepath = os.path.join(gold_feature_store_directory, partition_name)
        df.coalesce(1).write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath)
    finally:
        try:
            df.unpersist()
        except Exception:
            logging.warning("Could not release cached gold features", exc_info=True)

    return df
