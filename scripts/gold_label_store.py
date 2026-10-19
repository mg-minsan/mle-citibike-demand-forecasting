import argparse
import os

import pyspark

import utils.data_processing_gold_table


def main(snapshotdate):
    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_label_store") \
        .master("local[*]") \
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
        .getOrCreate()

    try:
        spark.sparkContext.setLogLevel("ERROR")
        silver_trips_directory = "datamart/silver/trips/"
        gold_store_directory = "datamart/gold/label_store/"
        os.makedirs(gold_store_directory, exist_ok=True)

        utils.data_processing_gold_table.process_labels_gold_table(
            snapshotdate, silver_trips_directory, gold_store_directory, spark,
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process daily gold_label_store")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
