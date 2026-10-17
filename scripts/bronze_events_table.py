import argparse
import os
import pyspark
from utils.data_processing_bronze_events_table import process_bronze_events_table

def main(snapshotdate):
    bronze_events_directory = "datamart/bronze/bronze_events"
    os.makedirs(bronze_events_directory, exist_ok=True)

    spark = pyspark.sql.SparkSession.builder \
        .appName("bronze_events_table") \
        .master("local[*]") \
        .getOrCreate()
    try:
        spark.sparkContext.setLogLevel("ERROR")
        print(f"Processing bronze events table for snapshot date: {snapshotdate}")
        process_bronze_events_table(snapshotdate, bronze_events_directory, spark)
        print(f"Finished processing bronze events table for snapshot date: {snapshotdate}")
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process bronze events table")
    parser.add_argument("--snapshotdate", type=str, required=True, help="Snapshot date for processing")
    args = parser.parse_args()
    main(args.snapshotdate)
