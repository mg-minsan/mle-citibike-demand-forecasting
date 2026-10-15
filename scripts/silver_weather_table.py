import argparse
import os

from pyspark.sql import SparkSession

from utils.data_processing_silver_weather_table import process_silver_weather_table


def main(snapshotdate):
    silver_weather_directory = "datamart/silver/silver_weather"
    os.makedirs(silver_weather_directory, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("silver_weather_table")
        .master("local[*]")
        .getOrCreate()
    )
    try:
        spark.sparkContext.setLogLevel("ERROR")
        print(f"Processing silver weather table for snapshot date: {snapshotdate}")
        process_silver_weather_table(
            snapshotdate,
            "datamart/bronze/bronze_weather",
            silver_weather_directory,
            spark,
        )
        print(f"Finished processing silver weather table for snapshot date: {snapshotdate}")
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process silver weather table")
    parser.add_argument("--snapshotdate", type=str, required=True, help="Snapshot date for processing")
    args = parser.parse_args()
    main(args.snapshotdate)
