import argparse
import os

from pyspark.sql import SparkSession

from utils.data_processing_gold_weather_table import process_gold_weather_table


def main(snapshotdate):
    gold_weather_directory = "datamart/gold/gold_weather"
    os.makedirs(gold_weather_directory, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("gold_weather_table")
        .master("local[*]")
        .getOrCreate()
    )
    try:
        spark.sparkContext.setLogLevel("ERROR")
        print(f"Processing gold weather table for snapshot date: {snapshotdate}")
        process_gold_weather_table(
            snapshotdate,
            "datamart/silver/silver_weather",
            gold_weather_directory,
            spark,
        )
        print(f"Finished processing gold weather table for snapshot date: {snapshotdate}")
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process gold weather table")
    parser.add_argument("--snapshotdate", type=str, required=True, help="Snapshot date for processing")
    args = parser.parse_args()
    main(args.snapshotdate)
