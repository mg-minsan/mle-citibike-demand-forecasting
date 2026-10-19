import os
import glob
from datetime import datetime

import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col


def find_trip_csv_files(snapshot_date):
    # raw files live in data/<year>-citibike-tripdata/<N_Month>/*.csv (some months have 2 csv files)
    year_folder = "data/" + str(snapshot_date.year) + "-citibike-tripdata/"
    month_folders = glob.glob(year_folder + str(snapshot_date.month) + "_*") + \
                    glob.glob(year_folder + str(snapshot_date.month).zfill(2) + "_*")

    csv_files = []
    for month_folder in sorted(set(month_folders)):
        for f in sorted(glob.glob(month_folder + "/*.csv")):
            name = os.path.basename(f)
            # skip Jersey City files (not part of NYC scope) and mac hidden files
            if name.startswith("JC") or name.startswith("._"):
                continue
            csv_files.append(f)
    return csv_files


def find_start_time_column(columns):
    # header changes across months: "starttime" (2016) vs "Start Time" (some 2017 months)
    for c in columns:
        if c.lower().replace(" ", "").replace("_", "") in ("starttime", "startedat"):
            return c
    raise ValueError("no start time column found in: " + str(columns))


def process_bronze_table(snapshot_date_str, bronze_trips_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # connect to source back end - IRL connect to back end source system
    csv_files = find_trip_csv_files(snapshot_date)
    if len(csv_files) == 0:
        raise FileNotFoundError("no trip csv found for " + snapshot_date_str)

    # load data - IRL ingest from back end source system
    df = None
    for csv_file_path in csv_files:
        # read everything as string: bronze keeps the raw values, silver does the typing/cleaning
        df_file = spark.read.csv(csv_file_path, header=True, inferSchema=False)
        start_col = find_start_time_column(df_file.columns)

        # keep only this day's trips. Two date formats exist: "1/31/2016 23:59:59" and "2016-10-01 00:00:06"
        start_day = F.split(col(start_col), " ").getItem(0)
        start_date = F.when(start_day.contains("/"), F.to_date(start_day, "M/d/yyyy")) \
                      .otherwise(F.to_date(start_day, "yyyy-MM-dd"))
        df_file = df_file.filter(start_date == F.lit(snapshot_date_str).cast("date"))

        df_file = df_file.withColumn("source_file", F.lit(os.path.basename(csv_file_path)))
        print("read:", csv_file_path, "| rows for", snapshot_date_str + ":", df_file.count())

        # month with 2 files: stack them, matching columns by name
        df = df_file if df is None else df.unionByName(df_file, allowMissingColumns=True)

    df = df.withColumn("snapshot_date", F.lit(snapshot_date_str).cast("date"))
    print(snapshot_date_str + " row count:", df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_trips_" + snapshot_date_str.replace('-', '_') + '.parquet'
    filepath = bronze_trips_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df
