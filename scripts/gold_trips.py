import argparse
import os

import pyspark

import utils.data_processing_gold_table

# to call this script (from inside scripts/): python gold_trips.py --snapshotdate "2016-02-05"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # create gold datalake
    silver_trips_directory = "datamart/silver/trips/"
    gold_label_store_directory = "datamart/gold/label_store/"
    gold_feature_store_directory = "datamart/gold/feature_store/"

    for directory in [gold_label_store_directory, gold_feature_store_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # run data processing
    utils.data_processing_gold_table.process_labels_gold_table(snapshotdate, silver_trips_directory, gold_label_store_directory, spark)
    utils.data_processing_gold_table.process_features_gold_table(snapshotdate, silver_trips_directory, gold_feature_store_directory, spark)

    # end spark session
    spark.stop()

    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    # Call main with arguments explicitly passed
    main(args.snapshotdate)
