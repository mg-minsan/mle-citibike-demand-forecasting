"""Validation for daily holiday and event partitions."""

import pyspark.sql.functions as F


HOLIDAY_COLUMNS = ["holiday_year", "holiday_date", "weekday", "observed_date", "holiday_name", "holiday_category"]
EVENT_COLUMNS = [
    "event_id", "event_name", "start_time", "end_time", "event_agency", "event_type",
    "event_borough", "event_location", "street_side", "street_closure_type",
    "community_board", "police_precinct", "cems_id",
]


def _require_columns(df, required, label):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


def validate_holidays(df, snapshot_date_str):
    """Reject null fields, weekday mismatches, duplicates, and rows unrelated to the day."""
    _require_columns(df, HOLIDAY_COLUMNS, "holiday")

    invalid = F.lit(False)
    for column in HOLIDAY_COLUMNS:
        invalid = invalid | F.col(column).isNull()
    if df.filter(invalid).limit(1).count():
        raise ValueError("Holidays contain null/invalid dates or names after casting.")

    if df.filter(F.date_format("holiday_date", "EEEE") != F.col("weekday")).limit(1).count():
        raise ValueError("Holiday weekday does not match holiday_date.")
    if df.filter(F.col("observed_date") < F.col("holiday_date")).limit(1).count():
        raise ValueError("Holiday observed_date precedes holiday_date.")
    if df.filter(~F.col("holiday_category").isin("federal", "nyc_calendar")).limit(1).count():
        raise ValueError("Unknown holiday_category.")
    if df.groupBy("holiday_date", "holiday_name").count().filter(F.col("count") > 1).limit(1).count():
        raise ValueError("Duplicate holidays in the requested day.")

    day = F.lit(snapshot_date_str).cast("date")
    if df.filter((F.col("holiday_date") != day) & (F.col("observed_date") != day)).limit(1).count():
        raise ValueError("Holidays contain rows unrelated to the requested day.")


def validate_events(df, snapshot_date_str):
    """Reject missing keys/timestamps, duplicate permit rows, and out-of-partition events.

    Rows whose end precedes their start exist in the source; Silver flags them
    (is_valid_time_range) rather than rejecting or dropping them.
    """
    _require_columns(df, EVENT_COLUMNS, "event")

    invalid = (
        F.col("event_id").isNull() | F.col("start_time").isNull() | F.col("end_time").isNull()
        | F.col("event_name").isNull() | F.col("event_type").isNull()
    )
    if df.filter(invalid).limit(1).count():
        raise ValueError("Events contain null/invalid ids, names, types or timestamps after casting.")

    if df.filter(F.to_date("start_time") != F.lit(snapshot_date_str).cast("date")).limit(1).count():
        raise ValueError("Events contain start times outside the requested day.")

    key = ["event_id", "start_time", "end_time", "event_location"]
    if df.groupBy(*key).count().filter(F.col("count") > 1).limit(1).count():
        raise ValueError("Duplicate event rows (event_id, start_time, end_time, event_location).")
