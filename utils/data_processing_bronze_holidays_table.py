import os
from datetime import datetime

import pyspark.sql.functions as F

from utils.utils import partition_path


def process_bronze_holidays_table(snapshot_date_str, bronze_holidays_directory, spark):
    datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Preserve source strings; type casting belongs in Silver. Each source file is tagged
    # with its category. A day is kept for the date it falls on and for the date it is
    # observed (e.g. a Sunday federal holiday moves to Monday).
    sources = {
        "federal": "data/us_federal_holidays_2016_2017.csv",
        "nyc_calendar": "data/nyc_calendar_days_2016_2017.csv",
    }
    frames = [
        spark.read.csv(path, header=True, inferSchema=False).withColumn("holiday_category", F.lit(category))
        for category, path in sources.items()
    ]
    sdf = frames[0]
    for frame in frames[1:]:
        sdf = sdf.unionByName(frame)
    sdf = sdf.filter(
        (F.col("date") == snapshot_date_str) | (F.col("observed_date") == snapshot_date_str)
    )
    print(f"Processing bronze holidays table for snapshot date: {snapshot_date_str}")
    print(f"Number of records: {sdf.count()}")

    os.makedirs(bronze_holidays_directory, exist_ok=True)
    filepath = partition_path(bronze_holidays_directory, "bronze_holidays_", snapshot_date_str, ".csv")
    sdf.toPandas().to_csv(filepath, index=False)
    print("saved to", filepath)

    return sdf
