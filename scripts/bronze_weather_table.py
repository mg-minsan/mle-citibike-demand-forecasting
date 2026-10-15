import argparse
import os
from dateutil.relativedelta import relativedelta
import pyspark
from utils.data_processing_bronze_weather_table import process_bronze_weather_table

def main(snapshotdate):
    bronze_weather_directory = "datamart/bronze/bronze_weather"
    os.makedirs(bronze_weather_directory, exist_ok=True)

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print(f"Processing bronze weather table for snapshot date: {snapshotdate}")
    date_str = snapshotdate
    process_bronze_weather_table(date_str, bronze_weather_directory, spark)
    print(f"Finished processing bronze weather table for snapshot date: {snapshotdate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process bronze weather table")
    parser.add_argument("--snapshotdate", type=str, help="Snapshot date for processing")
    args = parser.parse_args()
    main(args.snapshotdate)
