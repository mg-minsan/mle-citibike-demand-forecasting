import os
import argparse
from datetime import datetime, timedelta

import pyspark

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# bulk backfill of the history, one Spark session for all days (lab 2 main.py pattern)
# to call this script (from inside scripts/): python main.py --startdate 2016-01-01 --enddate 2017-12-31
# only one layer (e.g. bronze already done):  python main.py --startdate 2016-01-01 --enddate 2017-12-31 --layer silver
# gold needs the 28 days of silver before each date, so run gold after the silver backfill has finished
# Airflow later runs the same function one day at a time.


# generate list of dates to process
def generate_daily_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date = current_date + timedelta(days=1)
    return dates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="backfill history")
    parser.add_argument("--startdate", type=str, default="2016-01-01", help="YYYY-MM-DD")
    parser.add_argument("--enddate", type=str, default="2017-12-31", help="YYYY-MM-DD")
    parser.add_argument("--layer", type=str, default="all", help="all, bronze, silver or gold")
    args = parser.parse_args()

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    dates_str_lst = generate_daily_dates(args.startdate, args.enddate)
    print("days to process:", len(dates_str_lst), dates_str_lst[0], "to", dates_str_lst[-1])

    # create bronze datalake
    bronze_trips_directory = "datamart/bronze/trips/"

    if not os.path.exists(bronze_trips_directory):
        os.makedirs(bronze_trips_directory)

    # run bronze backfill
    if args.layer in ("all", "bronze"):
        for date_str in dates_str_lst:
            utils.data_processing_bronze_table.process_bronze_table(date_str, bronze_trips_directory, spark)

    # create silver datalake
    silver_trips_directory = "datamart/silver/trips/"

    if not os.path.exists(silver_trips_directory):
        os.makedirs(silver_trips_directory)

    # run silver backfill
    if args.layer in ("all", "silver"):
        for date_str in dates_str_lst:
            utils.data_processing_silver_table.process_silver_table(date_str, bronze_trips_directory, silver_trips_directory, spark)

    # create gold datalake
    gold_label_store_directory = "datamart/gold/label_store/"
    gold_feature_store_directory = "datamart/gold/feature_store/"

    for directory in [gold_label_store_directory, gold_feature_store_directory]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # run gold backfill
    if args.layer in ("all", "gold"):
        for date_str in dates_str_lst:
            utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_trips_directory, gold_label_store_directory, spark)
            utils.data_processing_gold_table.process_features_gold_table(date_str, silver_trips_directory, gold_feature_store_directory, spark)

    spark.stop()
