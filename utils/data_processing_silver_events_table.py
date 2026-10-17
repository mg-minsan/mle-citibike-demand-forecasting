import pyspark.sql.functions as F
from pyspark.sql.types import LongType

from utils.utils import read_csv_partition, write_parquet_partition
from utils.calendar_validation import validate_events


TIMESTAMP_FORMAT = "MM/dd/yyyy hh:mm:ss a"
# Permits running longer than this (e.g. season-long exhibitions) are flagged for Gold.
LONG_RUNNING_DAYS = 30


def _clean_text(column):
    """Trim, and turn empty or N/A placeholders into null."""
    trimmed = F.trim(F.col(column))
    return F.when((trimmed == "") | (F.upper(trimmed) == "N/A"), None).otherwise(trimmed)


def _clean_id_list(column):
    """Turn '5,' or '233, 232,' into '5' or '233,232'; empty and N/A become null."""
    cleaned = F.regexp_replace(F.regexp_replace(F.col(column), r"\s+", ""), r",+$", "")
    return F.when((cleaned == "") | (F.upper(cleaned) == "N/A"), None).otherwise(cleaned)


def process_silver_events_table(snapshot_date_str, bronze_events_directory, silver_events_directory, spark):
    df = read_csv_partition(
        bronze_events_directory, "bronze_events_", snapshot_date_str, spark,
        infer_schema=False,
    )

    column_name_map = {
        "Event ID": "event_id",
        "Event Name": "event_name",
        "Start Date/Time": "start_time",
        "End Date/Time": "end_time",
        "Event Agency": "event_agency",
        "Event Type": "event_type",
        "Event Borough": "event_borough",
        "Event Location": "event_location",
        "Event Street Side": "street_side",
        "Street Closure Type": "street_closure_type",
        "Community Board": "community_board",
        "Police Precinct": "police_precinct",
        "CEMSID": "cems_id",
    }
    missing_columns = sorted(set(column_name_map) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing source event columns: {missing_columns}")
    for original_name, new_name in column_name_map.items():
        df = df.withColumnRenamed(original_name, new_name)

    text_columns = [
        "event_name", "event_agency", "event_type", "event_borough", "event_location",
        "street_side", "street_closure_type",
    ]
    df = (
        df
        .withColumn("event_id", F.trim("event_id").cast(LongType()))
        # The CSV supplies local clock labels; timestamp_ntz avoids any DST shifting.
        .withColumn("start_time", F.to_timestamp_ntz(F.trim("start_time"), F.lit(TIMESTAMP_FORMAT)))
        .withColumn("end_time", F.to_timestamp_ntz(F.trim("end_time"), F.lit(TIMESTAMP_FORMAT)))
        .withColumn("community_board", _clean_id_list("community_board"))
        .withColumn("police_precinct", _clean_id_list("police_precinct"))
        .withColumn("cems_id", _clean_id_list("cems_id"))
    )
    for column in text_columns:
        df = df.withColumn(column, _clean_text(column))

    validate_events(df, snapshot_date_str)

    # Some source permits end before they start; flag them instead of dropping rows.
    valid_range = F.col("end_time") >= F.col("start_time")
    df = (
        df
        .withColumn("snapshot_date", F.to_date("start_time"))
        .withColumn("is_valid_time_range", valid_range)
        .withColumn(
            "duration_hours",
            # timestamp_ntz cannot be cast to long, so take the difference in SQL.
            F.when(valid_range, F.expr("timestampdiff(SECOND, start_time, end_time)") / 3600.0),
        )
        .withColumn(
            "is_long_running",
            valid_range & (F.col("duration_hours") > LONG_RUNNING_DAYS * 24),
        )
        .withColumn("is_multi_day", valid_range & (F.to_date("end_time") > F.to_date("start_time")))
    )
    # event_id repeats across rows (one per date/location), so build a deterministic row key.
    df = df.withColumn(
        "event_row_key",
        F.sha2(F.concat_ws(
            "|", F.col("event_id").cast("string"), F.col("start_time").cast("string"),
            F.col("end_time").cast("string"), F.coalesce(F.col("event_location"), F.lit("")),
        ), 256),
    )
    return write_parquet_partition(df, silver_events_directory, "silver_events_", snapshot_date_str)
