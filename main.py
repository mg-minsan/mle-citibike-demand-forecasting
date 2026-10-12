

# set up confi
from utils.utils import generate_first_of_month_dates
from utils.data_processing_bronze_weather_table import process_bronze_weather_table 
from utils.data_processing_silver_weather_table import  process_silver_weather_table
from utils.data_processing_gold_weather_table import process_gold_weather_table

import pyspark
import os


START_DATE_STR = "2016-01-01"
END_DATE_STR = "2017-12-01"


def main():
    first_of_month_dates = generate_first_of_month_dates(START_DATE_STR, END_DATE_STR)

    print(first_of_month_dates)
    spark = (pyspark.sql.SparkSession.builder
         .appName("data_ingestion_pipeline")
         .master("local[*]")
         .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    bronze_weather_directory = "datamart/bronze/weather/"
    if not os.path.exists(bronze_weather_directory):
        os.makedirs(bronze_weather_directory)

    silver_weather_directory = "datamart/silver/weather/"
    if not os.path.exists(silver_weather_directory):
        os.makedirs(silver_weather_directory)

    gold_weather_directory = "datamart/gold/weather/"
    if not os.path.exists(gold_weather_directory):
        os.makedirs(gold_weather_directory)

    for snapshot_date_str in first_of_month_dates:
        process_bronze_weather_table(snapshot_date_str, bronze_weather_directory, spark)
        process_silver_weather_table(snapshot_date_str, bronze_weather_directory, silver_weather_directory, spark)
        process_gold_weather_table(snapshot_date_str, silver_weather_directory, gold_weather_directory, spark)




if __name__ == "__main__":
    main()