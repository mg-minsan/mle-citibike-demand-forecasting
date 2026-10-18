import argparse
import os

import pyspark

import utils.data_processing_silver_table

# to call this script (from inside scripts/): python silver_trips.py --snapshotdate "2016-01-05"

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

    # create silver datalake
    bronze_trips_directory = "datamart/bronze/trips/"
    silver_trips_directory = "datamart/silver/trips/"

    if not os.path.exists(silver_trips_directory):
        os.makedirs(silver_trips_directory)

    # run data processing
    utils.data_processing_silver_table.process_silver_table(snapshotdate, bronze_trips_directory, silver_trips_directory, spark)

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
