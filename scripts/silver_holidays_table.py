import argparse
import os

from pyspark.sql import SparkSession

from utils.data_processing_silver_holidays_table import process_silver_holidays_table


def main(snapshotdate):
    silver_holidays_directory = "datamart/silver/silver_holidays"
    os.makedirs(silver_holidays_directory, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("silver_holidays_table")
        .master("local[*]")
        .getOrCreate()
    )
    try:
        spark.sparkContext.setLogLevel("ERROR")
        print(f"Processing silver holidays table for snapshot date: {snapshotdate}")
        process_silver_holidays_table(
            snapshotdate,
            "datamart/bronze/bronze_holidays",
            silver_holidays_directory,
            spark,
        )
        print(f"Finished processing silver holidays table for snapshot date: {snapshotdate}")
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process silver holidays table")
    parser.add_argument("--snapshotdate", type=str, required=True, help="Snapshot date for processing")
    args = parser.parse_args()
    main(args.snapshotdate)
