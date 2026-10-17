import argparse
import os

from pyspark.sql import SparkSession

from utils.data_processing_silver_events_table import process_silver_events_table


def main(snapshotdate):
    silver_events_directory = "datamart/silver/silver_events"
    os.makedirs(silver_events_directory, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("silver_events_table")
        .master("local[*]")
        .getOrCreate()
    )
    try:
        spark.sparkContext.setLogLevel("ERROR")
        print(f"Processing silver events table for snapshot date: {snapshotdate}")
        process_silver_events_table(
            snapshotdate,
            "datamart/bronze/bronze_events",
            silver_events_directory,
            spark,
        )
        print(f"Finished processing silver events table for snapshot date: {snapshotdate}")
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process silver events table")
    parser.add_argument("--snapshotdate", type=str, required=True, help="Snapshot date for processing")
    args = parser.parse_args()
    main(args.snapshotdate)
