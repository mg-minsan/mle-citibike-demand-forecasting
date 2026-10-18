"""Daily trips and weather pipeline, followed by gold labels and joined features.

Run from the repository root:
    python main.py --startdate 2016-01-01 --enddate 2017-12-31
"""
import argparse
import logging
import os
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pyspark

from utils.utils import generate_daily_dates
from utils.data_processing_bronze_weather_table import process_bronze_weather_table
from utils.data_processing_silver_weather_table import process_silver_weather_table
from utils.data_processing_bronze_table import process_bronze_table
from utils.data_processing_silver_table import process_silver_table
from utils.data_processing_gold_table import process_labels_gold_table, process_features_gold_table


START_DATE_STR = "2016-01-01"
END_DATE_STR = "2017-12-31"
WEATHER_SOURCE_TIMEZONE = "America/New_York"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build daily trip, weather, label and feature partitions.")
    parser.add_argument("--startdate", default=START_DATE_STR, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--enddate", default=END_DATE_STR, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--layer", choices=("all", "bronze", "silver", "gold"), default="all")
    parser.add_argument(
        "--timezone",
        default=os.environ.get("WEATHER_SOURCE_TIMEZONE") or WEATHER_SOURCE_TIMEZONE,
        help="Source timezone (default: America/New_York); overrides WEATHER_SOURCE_TIMEZONE.",
    )
    args = parser.parse_args(argv)
    try:
        ZoneInfo(args.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        parser.error(f"Invalid source timezone: {args.timezone}")
    try:
        args.dates = generate_daily_dates(args.startdate, args.enddate)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ["WEATHER_SOURCE_TIMEZONE"] = args.timezone
    print("days to process:", len(args.dates), args.dates[0], "to", args.dates[-1])

    bronze_weather_directory = "datamart/bronze/weather/"
    silver_weather_directory = "datamart/silver/weather/"
    bronze_trips_directory = "datamart/bronze/trips/"
    silver_trips_directory = "datamart/silver/trips/"
    gold_label_store_directory = "datamart/gold/label_store/"
    gold_feature_store_directory = "datamart/gold/feature_store/"
    for directory in (bronze_weather_directory, silver_weather_directory,
                      bronze_trips_directory, silver_trips_directory,
                      gold_label_store_directory, gold_feature_store_directory):
        os.makedirs(directory, exist_ok=True)

    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 4g pyspark-shell")
    spark = (pyspark.sql.SparkSession.builder
             .appName("daily_data_pipeline")
             .master("local[2]")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))
             .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
             .config("spark.sql.session.timeZone", args.timezone)
             .getOrCreate())
    try:
        spark.sparkContext.setLogLevel("ERROR")
        # Finish each day before advancing, so gold can read prior silver history.
        for day in args.dates:
            if args.layer in ("all", "bronze"):
                process_bronze_weather_table(day, bronze_weather_directory, spark)
                process_bronze_table(day, bronze_trips_directory, spark)
            if args.layer in ("all", "silver"):
                process_silver_weather_table(day, bronze_weather_directory, silver_weather_directory, spark)
                process_silver_table(day, bronze_trips_directory, silver_trips_directory, spark)
            if args.layer in ("all", "gold"):
                process_labels_gold_table(day, silver_trips_directory, gold_label_store_directory, spark)
                process_features_gold_table(day, silver_trips_directory, gold_feature_store_directory, spark,
                                            silver_weather_directory=silver_weather_directory)
    except BaseException:
        try:
            spark.stop()
        except Exception:
            logging.warning("Could not stop Spark after processing failed", exc_info=True)
        raise
    else:
        spark.stop()


if __name__ == "__main__":
    main()
