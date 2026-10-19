import argparse
import glob
from datetime import datetime

import pyspark

import utils.data_processing_bronze_table

# data check: do one month's daily bronze files add up to the raw CSV row count?
# to call this script (from inside scripts/): python check_bronze_month.py --month 2016-01

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="check bronze month")
    parser.add_argument("--month", type=str, required=True, help="YYYY-MM")
    args = parser.parse_args()

    # raw rows: count lines in the month's CSV(s), minus 1 header line per file
    month_date = datetime.strptime(args.month + "-01", "%Y-%m-%d")
    raw_count = 0
    for f in utils.data_processing_bronze_table.find_trip_csv_files(month_date):
        with open(f, encoding="utf-8-sig") as fh:
            raw_count += sum(1 for line in fh if line.strip()) - 1

    # bronze rows: read all daily bronze files of that month
    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    files_list = glob.glob("datamart/bronze/trips/bronze_trips_" + args.month.replace('-', '_') + "_*.parquet")
    bronze_count = spark.read.parquet(*files_list).count()
    spark.stop()

    print("daily bronze files:", len(files_list))
    print("raw csv rows:      ", raw_count)
    print("bronze rows:       ", bronze_count)
    print("MATCH" if raw_count == bronze_count else "MISMATCH")
