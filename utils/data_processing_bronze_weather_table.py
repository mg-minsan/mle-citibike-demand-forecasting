import os
from datetime import datetime

import pyspark.sql.functions as F

from utils.utils import partition_path
from utils.weather_validation import configure_weather_timezone


def process_bronze_weather_table(snapshot_date_str, bronze_weather_directory, spark):
    configure_weather_timezone(spark)
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    snapshot_day = snapshot_date.strftime("%Y-%m-%d")

    csv_file_path = "data/NYC_Weather_2016_2022.csv"
    # Preserve source strings; timestamp parsing belongs in Silver.
    sdf = spark.read.csv(csv_file_path, header=True, inferSchema=False).filter(
        F.col("time").startswith(snapshot_day)
    )
    print(f"Processing bronze weather table for snapshot date: {snapshot_date_str}")
    print(f"Number of records: {sdf.count()}")

    os.makedirs(bronze_weather_directory, exist_ok=True)
    filepath = partition_path(bronze_weather_directory, "bronze_weather_", snapshot_date_str, ".csv")
    sdf.toPandas().to_csv(filepath, index=False)
    print("saved to", filepath)

    return sdf