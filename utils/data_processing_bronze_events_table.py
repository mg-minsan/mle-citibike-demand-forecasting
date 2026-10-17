import os
from datetime import datetime

import pyspark.sql.functions as F

from utils.utils import partition_path


def process_bronze_events_table(snapshot_date_str, bronze_events_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    # Source timestamps look like "01/31/2016 09:00:00 AM"; match on the MM/dd/yyyy prefix.
    snapshot_prefix = snapshot_date.strftime("%m/%d/%Y")

    csv_file_path = "data/NYC_Permitted_Event_Information_-_Historical_20260927.csv"
    # Preserve source strings; the source escapes quotes by doubling them. Events are
    # partitioned by start date, so multi-day permits stay on their start-date partition.
    sdf = (
        spark.read.csv(csv_file_path, header=True, inferSchema=False, quote='"', escape='"')
        .filter(F.col("Start Date/Time").startswith(snapshot_prefix))
    )
    print(f"Processing bronze events table for snapshot date: {snapshot_date_str}")
    print(f"Number of records: {sdf.count()}")

    os.makedirs(bronze_events_directory, exist_ok=True)
    filepath = partition_path(bronze_events_directory, "bronze_events_", snapshot_date_str, ".csv")
    sdf.toPandas().to_csv(filepath, index=False)
    print("saved to", filepath)

    return sdf
