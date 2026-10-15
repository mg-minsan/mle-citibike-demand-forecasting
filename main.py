

# set up confi
from utils.utils import generate_daily_dates
from utils.data_processing_bronze_weather_table import process_bronze_weather_table 
from utils.data_processing_silver_weather_table import  process_silver_weather_table
from utils.data_processing_gold_weather_table import process_gold_weather_table

import argparse
import os
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pyspark


START_DATE_STR = "2016-01-01"
END_DATE_STR = "2017-12-31"
WEATHER_SOURCE_TIMEZONE = "America/New_York"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Ingest daily weather partitions.")
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
    return args


def main(argv=None):
    args = parse_args(argv)
    os.environ["WEATHER_SOURCE_TIMEZONE"] = args.timezone
    daily_dates = generate_daily_dates(START_DATE_STR, END_DATE_STR)

    print(daily_dates)
    spark = (pyspark.sql.SparkSession.builder
         .appName("data_ingestion_pipeline")
         .master("local[*]")
         .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    bronze_weather_directory = "datamart/bronze/weather/"
    if not os.path.exists(bronze_weather_directory):
        os.makedirs(bronze_weather_directory)

    silver_weather_directory = "datamart/silver/weather/"
    if not os.path.exists(silver_weather_directory):
        os.makedirs(silver_weather_directory)

    gold_weather_directory = "datamart/gold/weather/"
    if not os.path.exists(gold_weather_directory):
        os.makedirs(gold_weather_directory)

    for snapshot_date_str in daily_dates:
        process_bronze_weather_table(snapshot_date_str, bronze_weather_directory, spark)
        process_silver_weather_table(snapshot_date_str, bronze_weather_directory, silver_weather_directory, spark)
        process_gold_weather_table(snapshot_date_str, silver_weather_directory, gold_weather_directory, spark)




if __name__ == "__main__":
    main()