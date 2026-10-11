import os
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pyspark.sql.functions as F


def process_bronze_weather_table(snapshot_date_str, bronze_weather_directory, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    year_month = snapshot_date.strftime("%Y-%m")

    csv_file_apth = "data/NYC_Weather_2016_2022.csv"
    sdf = spark.read.csv(csv_file_apth, header=True, inferSchema=True).filter(
        F.col("time").startswith(year_month)
    )
    print(f"Processing bronze weather table for snapshot date: {snapshot_date_str}")
    print(f"Number of records: {sdf.count()}")

    partition_name = "bronze_weather_" + snapshot_date_str.replace("-", "_") + ".csv"
    filepath = bronze_weather_directory + partition_name
    sdf.toPandas().to_csv(filepath, index=False)
    print("saved to", filepath)

    return sdf