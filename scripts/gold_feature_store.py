import argparse
import os

import pyspark

import utils.data_processing_gold_table


def main(snapshotdate):
    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_feature_store") \
        .master("local[*]") \
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
        .getOrCreate()

    try:
        spark.sparkContext.setLogLevel("ERROR")
        silver_trips_directory = "datamart/silver/trips/"
        silver_weather_directory = "datamart/silver/weather/"
        gold_store_directory = "datamart/gold/feature_store/"
        os.makedirs(gold_store_directory, exist_ok=True)

        utils.data_processing_gold_table.process_features_gold_table(
            snapshotdate, silver_trips_directory, gold_store_directory, spark,
            silver_weather_directory=silver_weather_directory,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process daily gold_feature_store")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
