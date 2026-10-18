import os
from datetime import datetime

import pyspark
from pyspark import StorageLevel
from utils_monthly.utils import month_bounds, monthly_partition_path
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, DoubleType, TimestampType


# raw header (lower case, spaces removed) -> clean column name
# covers both header styles: "tripduration,starttime,..." and "Trip Duration,Start Time,..."
column_name_map = {
    "tripduration": "trip_duration",
    "starttime": "start_time",
    "stoptime": "stop_time",
    "startstationid": "start_station_id",
    "startstationname": "start_station_name",
    "startstationlatitude": "start_station_latitude",
    "startstationlongitude": "start_station_longitude",
    "endstationid": "end_station_id",
    "endstationname": "end_station_name",
    "endstationlatitude": "end_station_latitude",
    "endstationlongitude": "end_station_longitude",
    "bikeid": "bike_id",
    "usertype": "user_type",
    "birthyear": "birth_year",
    "gender": "gender",
}

# clean column name -> data type (start/stop time are parsed separately below)
column_type_map = {
    "trip_duration": IntegerType(),
    "start_station_id": StringType(),
    "start_station_name": StringType(),
    "start_station_latitude": DoubleType(),
    "start_station_longitude": DoubleType(),
    "end_station_id": StringType(),
    "end_station_name": StringType(),
    "end_station_latitude": DoubleType(),
    "end_station_longitude": DoubleType(),
    "bike_id": StringType(),
    "user_type": StringType(),
    "birth_year": IntegerType(),
    "gender": IntegerType(),
}


# station names that are Citi Bike depots, warehouses, test or ops stations, not customer stations (found in the EDA,
# e.g. "NYCBS Depot - SSP", "8D OPS 01", "333 Johnson TEST 1", "Penn Station Valet - Valet Scan")
non_customer_station_pattern = r"(?i)depot|warehouse|\btest\b|^8d |kiosk in a box|valet scan"

# rough box around New York City; stations outside it are wrong coordinates (0,0 or test stations in Montreal etc.)
nyc_latitude_range = (40.4, 41.0)
nyc_longitude_range = (-74.3, -73.6)


def parse_trip_timestamp(c):
    # two formats in the raw files: "1/31/2016 23:59:59" (Jan-Sep 2016) and "2016-10-01 00:00:06" (later)
    return F.when(c.contains("/"), F.to_timestamp(c, "M/d/yyyy H:mm:ss")) \
            .otherwise(F.to_timestamp(F.substring(c, 1, 19), "yyyy-MM-dd HH:mm:ss"))


def process_silver_table(snapshot_date_str, bronze_trips_directory, silver_trips_directory, spark):
    # connect to bronze table
    filepath = monthly_partition_path(bronze_trips_directory, "bronze_trips", snapshot_date_str)
    df = spark.read.parquet(filepath)
    month_start, month_end = month_bounds(snapshot_date_str)
    df = df.filter((col("snapshot_date") >= F.lit(month_start)) & (col("snapshot_date") < F.lit(month_end)))
    rows_in = df.count()
    print('loaded from:', filepath, 'row count:', rows_in)

    # clean data: rename both header styles to one set of names
    # (if a day ever has both styles, coalesce takes whichever one is filled)
    new_columns = []
    for raw_key, clean_name in column_name_map.items():
        matches = [c for c in df.columns if c.lower().replace(" ", "") == raw_key]
        if len(matches) > 0:
            new_columns.append(F.coalesce(*[col("`" + c + "`") for c in matches]).alias(clean_name))
        else:
            new_columns.append(F.lit(None).cast(StringType()).alias(clean_name))
    df = df.select(*new_columns, col("source_file"), col("snapshot_date"))

    # clean data: enforce schema / data type
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, F.trim(col(column)).cast(new_type))
    df = df.withColumn("start_time", parse_trip_timestamp(col("start_time")).cast(TimestampType()))
    df = df.withColumn("stop_time", parse_trip_timestamp(col("stop_time")).cast(TimestampType()))

    # clean data: drop rows we can't use, and print how many each check removed
    non_customer_station = F.coalesce(col("start_station_name").rlike(non_customer_station_pattern), F.lit(False)) | \
                           F.coalesce(col("end_station_name").rlike(non_customer_station_pattern), F.lit(False))

    outside_nyc = F.lit(False)
    for c in ["start_station_latitude", "end_station_latitude"]:
        outside_nyc = outside_nyc | col(c).isNull() | ~col(c).between(*nyc_latitude_range)
    for c in ["start_station_longitude", "end_station_longitude"]:
        outside_nyc = outside_nyc | col(c).isNull() | ~col(c).between(*nyc_longitude_range)

    checks = [
        ("missing start_time", col("start_time").isNull()),
        ("missing start or end station id", col("start_station_id").isNull() | col("end_station_id").isNull()),
        ("start_time not on snapshot date", F.to_date(col("start_time")) != col("snapshot_date")),
        ("trip_duration missing, under 60s or over 24h", col("trip_duration").isNull() | (col("trip_duration") < 60) | (col("trip_duration") > 24 * 60 * 60)),
        ("depot / test / warehouse station", non_customer_station),
        ("station coordinates missing, zero or outside NYC", outside_nyc),
    ]
    for reason, bad_row in checks:
        n_bad = df.filter(bad_row).count()
        df = df.filter(~bad_row)
        print('  dropped', n_bad, 'rows:', reason)

    rows_before_dedup = df.count()
    df = df.dropDuplicates()
    print('  dropped', rows_before_dedup - df.count(), 'rows: exact duplicates')

    # keep only the columns the project needs (rider details like birth year, gender, bike id are dropped)
    keep_columns = ["trip_duration", "start_time", "stop_time",
                    "start_station_id", "start_station_name", "start_station_latitude", "start_station_longitude",
                    "end_station_id", "end_station_name", "end_station_latitude", "end_station_longitude",
                    "source_file", "snapshot_date"]
    df = df.select(*keep_columns)

    # augment data: date and hour of the trip start (used by gold)
    df = df.withColumn("start_date", F.to_date(col("start_time")))
    df = df.withColumn("start_hour", F.hour(col("start_time")))

    df = df.persist(StorageLevel.DISK_ONLY)
    try:
        print(snapshot_date_str, 'rows in:', rows_in, '| rows out:', df.count())
        filepath = monthly_partition_path(silver_trips_directory, "silver_trips", snapshot_date_str)
        df.write.mode("overwrite").parquet(filepath)
        print('saved to:', filepath)
    finally:
        df.unpersist()
    return df
