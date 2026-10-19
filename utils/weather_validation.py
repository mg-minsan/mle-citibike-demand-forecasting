"""Timezone configuration and validation for daily hourly weather partitions."""

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pyspark.sql.functions as F
from pyspark.sql.types import TimestampNTZType


WEATHER_MEASUREMENTS = [
    "temperature_c", "precipitation_mm", "rain_mm", "cloud_pct",
    "cloud_low_pct", "cloud_mid_pct", "cloud_high_pct",
    "wind_speed_kmh", "wind_direction_deg",
]


def configure_weather_timezone(spark):
    """Configure the source timezone, defaulting to America/New_York.

    The CSV's local-clock timestamps use timestamp_ntz, so this setting does
    not convert or DST-adjust them. Gold preserves their supplied dates.
    """
    timezone = os.environ.get("WEATHER_SOURCE_TIMEZONE") or "America/New_York"
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Invalid WEATHER_SOURCE_TIMEZONE: {timezone}") from exc
    spark.conf.set("spark.sql.session.timeZone", timezone)
    return timezone


def validate_hourly_weather(df, snapshot_date_str, timezone):
    """Reject missing, duplicate, off-hour, and out-of-partition observations.

    Local-clock (timestamp_ntz) inputs require 24 distinct hourly labels per
    supplied date, without DST adjustment. Absolute timestamps retain DST-aware
    coverage checks. Raise before publishing incomplete Silver or Gold outputs.
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
    # Inspect clock labels without implicitly casting timestamp_ntz to an instant.
    whole_hour = F.col("time").cast("string").rlike(r"^\d{4}-\d{2}-\d{2} \d{2}:00:00$")
    if df.filter(~whole_hour).limit(1).count():
        raise ValueError("Weather timestamps must align to whole hours.")
    local_clock = isinstance(df.schema["time"].dataType, TimestampNTZType)
    if df.groupBy("time").count().filter(F.col("count") > 1).limit(1).count():
        if local_clock:
            raise ValueError("Duplicate hourly weather timestamps in the supplied local clock.")
        raise ValueError(
            "Duplicate hourly weather timestamps. Timestamp casting can create duplicates "
            "when the configured source timezone has a DST gap (e.g. New York 02:00 "
            "becomes 03:00). Verify WEATHER_SOURCE_TIMEZONE / --timezone against the "
            "source metadata; do not drop rows to bypass this check."
        )

    snapshot = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    if local_clock:
        outside_day = F.to_date("time") != F.lit(snapshot.date())
        expected_hours = 24
    else:
        # Absolute instants distinguish repeated local hours during DST fallback.
        start = snapshot.replace(tzinfo=ZoneInfo(timezone))
        end = start + timedelta(days=1)
        start_epoch, end_epoch = int(start.timestamp()), int(end.timestamp())
        epoch = F.col("time").cast("long")
        outside_day = (epoch < start_epoch) | (epoch >= end_epoch)
        expected_hours = (end_epoch - start_epoch) // 3600
    if df.filter(outside_day).limit(1).count():
        raise ValueError("Weather contains timestamps outside the requested day.")

    expected = [(snapshot.date(), expected_hours)]
    expected_df = df.sparkSession.createDataFrame(expected, "snapshot_date date, expected_hours long")
    actual = df.groupBy(F.to_date("time").alias("snapshot_date")).agg(
        F.count("*").alias("hourly_rows")
    )
    incomplete = expected_df.join(actual, "snapshot_date", "left").filter(
        F.col("hourly_rows").isNull() | (F.col("hourly_rows") != F.col("expected_hours"))
    )
    if incomplete.limit(1).count():
        raise ValueError("Incomplete hourly weather coverage in the requested day.")


def validate_weather_dates(df, expected_dates, timezone):
    """Validate a batch of complete daily partitions with one aggregation.

    Missing files are handled by the caller; existing empty, malformed,
    duplicate or off-date daily files must not silently reduce coverage.
    """
    missing = sorted({'time', *WEATHER_MEASUREMENTS} - set(df.columns))
    if missing:
        raise ValueError(f'Missing weather columns: {missing}')
    invalid = F.col('time').isNull() | ~F.col('time').cast('string').rlike(
        r'^\d{4}-\d{2}-\d{2} \d{2}:00:00$')
    for name in WEATHER_MEASUREMENTS:
        value = F.col(name)
        invalid = invalid | value.isNull() | F.isnan(value) | (F.abs(value) == float('inf'))
    local_clock = isinstance(df.schema['time'].dataType, TimestampNTZType)
    expected = {}
    for day in expected_dates:
        start = datetime.combine(day, datetime.min.time(), tzinfo=ZoneInfo(timezone))
        end = start + timedelta(days=1)
        expected[day] = 24 if local_clock else int((end.timestamp() - start.timestamp()) // 3600)
    rows = df.groupBy(F.to_date('time').alias('day')).agg(
        F.count('*').alias('hours'), F.countDistinct('time').alias('distinct_hours'),
        F.count(F.when(invalid, 1)).alias('invalid_rows'),
    ).collect()
    if (set(row.day for row in rows) != set(expected) or
            any(row.hours != expected[row.day] or row.distinct_hours != expected[row.day]
                or row.invalid_rows for row in rows)):
        raise ValueError('Invalid or incomplete hourly weather coverage in daily history partitions')
