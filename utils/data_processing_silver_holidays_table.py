import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType

from utils.utils import read_csv_partition, write_parquet_partition
from utils.calendar_validation import validate_holidays


def process_silver_holidays_table(snapshot_date_str, bronze_holidays_directory, silver_holidays_directory, spark):
    df = read_csv_partition(
        bronze_holidays_directory, "bronze_holidays_", snapshot_date_str, spark,
        infer_schema=False,
    )

    column_name_map = {
        "year": "holiday_year",
        "date": "holiday_date",
        "weekday": "weekday",
        "observed_date": "observed_date",
        "holiday_name": "holiday_name",
        "holiday_category": "holiday_category",
    }
    missing_columns = sorted(set(column_name_map) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing source holiday columns: {missing_columns}")
    for original_name, new_name in column_name_map.items():
        df = df.withColumnRenamed(original_name, new_name)

    df = (
        df
        .withColumn("holiday_year", F.col("holiday_year").cast(IntegerType()))
        .withColumn("holiday_date", F.to_date("holiday_date", "yyyy-MM-dd"))
        .withColumn("observed_date", F.to_date("observed_date", "yyyy-MM-dd"))
        .withColumn("weekday", F.trim("weekday"))
        .withColumn("holiday_name", F.trim("holiday_name"))
        .withColumn("holiday_category", F.trim("holiday_category"))
    )

    validate_holidays(df, snapshot_date_str)

    day = F.lit(snapshot_date_str).cast("date")
    df = (
        df
        .withColumn("snapshot_date", day)
        # is_holiday_date: the calendar date itself; is_observed_date: the day off in lieu.
        .withColumn("is_holiday_date", F.col("holiday_date") == day)
        .withColumn("is_observed_date", F.col("observed_date") == day)
        .withColumn("is_federal_holiday", F.col("holiday_category") == "federal")
        .withColumn("is_shifted", F.col("holiday_date") != F.col("observed_date"))
    )
    return write_parquet_partition(df, silver_holidays_directory, "silver_holidays_", snapshot_date_str)
