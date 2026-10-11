"""Timezone configuration and validation for monthly hourly weather partitions."""

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pyspark.sql.functions as F


WEATHER_MEASUREMENTS = [
    "temperature_c", "precipitation_mm", "rain_mm", "cloud_pct",
    "cloud_low_pct", "cloud_mid_pct", "cloud_high_pct",
    "wind_speed_kmh", "wind_direction_deg",
]


def configure_weather_timezone(spark):
    """Use a verified source timezone, never the machine's implicit timezone.

    Daily Gold dates use this same timezone. If NYC dates differ from the
    source dates, conversion and cross-month boundary reads are required.
    """
    timezone = os.environ.get("WEATHER_SOURCE_TIMEZONE")
    if not timezone:
        raise ValueError(
            "Set WEATHER_SOURCE_TIMEZONE to the verified source timezone "
            "(e.g. UTC or America/New_York); the CSV does not identify it."
        )
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Invalid WEATHER_SOURCE_TIMEZONE: {timezone}") from exc
    spark.conf.set("spark.sql.session.timeZone", timezone)
    return timezone


def validate_hourly_weather(df, snapshot_date_str, timezone):
    """Reject missing, duplicate, off-hour, and out-of-partition observations.

    Verify the complete month, including entirely missing days and DST days.
    Raise before publishing incomplete Silver or Gold outputs.
    """
    required = ["time", *WEATHER_MEASUREMENTS]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing weather columns: {missing}")

    invalid = F.col("time").isNull()
    for column in WEATHER_MEASUREMENTS:
        value = F.col(column)
        invalid = invalid | value.isNull() | F.isnan(value) | (F.abs(value) == float("inf"))
    if df.filter(invalid).limit(1).count():
        raise ValueError("Weather contains null/invalid timestamps or measurements after casting.")
    if df.filter(F.col("time") != F.date_trunc("hour", "time")).limit(1).count():
        raise ValueError("Weather timestamps must align to whole hours.")
    if df.groupBy("time").count().filter(F.col("count") > 1).limit(1).count():
        raise ValueError("Duplicate hourly weather timestamps.")

    # Absolute instants distinguish repeated local hours during the DST fallback.
    snapshot = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    start = datetime(snapshot.year, snapshot.month, 1, tzinfo=ZoneInfo(timezone))
    if snapshot.month == 12:
        end = datetime(snapshot.year + 1, 1, 1, tzinfo=ZoneInfo(timezone))
    else:
        end = datetime(snapshot.year, snapshot.month + 1, 1, tzinfo=ZoneInfo(timezone))
    start_epoch, end_epoch = int(start.timestamp()), int(end.timestamp())
    epoch = F.col("time").cast("long")
    if df.filter((epoch < start_epoch) | (epoch >= end_epoch)).limit(1).count():
        raise ValueError("Weather contains timestamps outside the requested month.")

    expected = []
    day = start
    while day < end:
        next_day = day + timedelta(days=1)
        expected.append((day.date(), int((next_day.timestamp() - day.timestamp()) / 3600)))
        day = next_day
    expected_df = df.sparkSession.createDataFrame(expected, "snapshot_date date, expected_hours long")
    actual = df.groupBy(F.to_date("time").alias("snapshot_date")).agg(
        F.count("*").alias("hourly_rows")
    )
    incomplete = expected_df.join(actual, "snapshot_date", "left").filter(
        F.col("hourly_rows").isNull() | (F.col("hourly_rows") != F.col("expected_hours"))
    )
    if incomplete.limit(1).count():
        raise ValueError("Incomplete daily weather coverage in the requested month (DST-aware).")
