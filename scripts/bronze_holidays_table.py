import argparse
import os
import pyspark
from utils.data_processing_bronze_holidays_table import process_bronze_holidays_table

def main(snapshotdate):
    bronze_holidays_directory = "datamart/bronze/bronze_holidays"
    os.makedirs(bronze_holidays_directory, exist_ok=True)

    spark = pyspark.sql.SparkSession.builder \
        .appName("bronze_holidays_table") \
        .master("local[*]") \
        .getOrCreate()
    try:
        spark.sparkContext.setLogLevel("ERROR")
        print(f"Processing bronze holidays table for snapshot date: {snapshotdate}")
        process_bronze_holidays_table(snapshotdate, bronze_holidays_directory, spark)
        print(f"Finished processing bronze holidays table for snapshot date: {snapshotdate}")
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process bronze holidays table")
    parser.add_argument("--snapshotdate", type=str, required=True, help="Snapshot date for processing")
    args = parser.parse_args()
    main(args.snapshotdate)
