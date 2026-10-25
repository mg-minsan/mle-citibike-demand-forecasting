import argparse
import os
from datetime import datetime, timedelta

import pyspark

import utils.data_processing_gold_table
from utils.utils import generate_daily_dates


def main(snapshotdate, enddate=None, missing_only=False):
    dates = generate_daily_dates(snapshotdate, enddate or snapshotdate)
    silver_trips_directory = "datamart/silver/trips/"
    gold_store_directory = "datamart/gold/label_store/"
    if missing_only:
        dates = [day for day in dates if not os.path.isfile(os.path.join(
            gold_store_directory, "gold_label_store_" + day.replace("-", "_") + ".parquet", "_SUCCESS"))]
    print("label partitions to build:", len(dates), flush=True)
    if not dates:
        return

    # The label builder treats absent silver days as zero. Backfills must have
    # the full source history so missing inputs cannot silently become labels.
    # Single-day runs retain their existing partial-history warm-up behavior.
    required = set()
    for day in dates if enddate is not None else []:
        snapshot = datetime.strptime(day, "%Y-%m-%d")
        for offset in range(utils.data_processing_gold_table.ACTIVE_WINDOW_DAYS + 1):
            source_day = snapshot - timedelta(days=offset)
            required.add(os.path.join(silver_trips_directory,
                                     "silver_trips_" + source_day.strftime("%Y_%m_%d") + ".parquet", "_SUCCESS"))
    missing = sorted(path for path in required if not os.path.isfile(path))
    if missing:
        raise FileNotFoundError(f"{len(missing)} silver partitions missing or incomplete, first: {missing[0]}")

    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 4g pyspark-shell")
    spark = pyspark.sql.SparkSession.builder \
        .appName("gold_label_store") \
        .master("local[2]") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED") \
        .getOrCreate()

    try:
        spark.sparkContext.setLogLevel("ERROR")
        os.makedirs(gold_store_directory, exist_ok=True)

        for day in dates:
            utils.data_processing_gold_table.process_labels_gold_table(
                day, silver_trips_directory, gold_store_directory, spark,
            )
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process daily gold_label_store")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--snapshotdate", help="Single day, YYYY-MM-DD")
    selection.add_argument("--startdate", help="First day of a backfill, YYYY-MM-DD")
    parser.add_argument("--enddate", help="Last day of a backfill, inclusive, YYYY-MM-DD")
    parser.add_argument("--missing-only", action="store_true", help="Skip partitions with _SUCCESS")
    args = parser.parse_args()
    if bool(args.startdate) != bool(args.enddate):
        parser.error("--startdate and --enddate must be supplied together")
    main(args.snapshotdate or args.startdate, args.enddate, args.missing_only)
