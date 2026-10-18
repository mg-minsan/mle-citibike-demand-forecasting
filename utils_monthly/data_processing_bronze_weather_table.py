"""Store raw weather strings in one Parquet partition per month."""
import pyspark.sql.functions as F

from utils.weather_validation import configure_weather_timezone
from utils_monthly.utils import month_bounds, write_parquet_partition

WEATHER_CSV_PATH = 'data/NYC_Weather_2016_2022.csv'


def process_bronze_weather_table(snapshot_date_str, bronze_weather_directory, spark):
    configure_weather_timezone(spark)
    start, end = month_bounds(snapshot_date_str)
    df = spark.read.csv(WEATHER_CSV_PATH, header=True, inferSchema=False)
    day = F.substring('time', 1, 10).cast('date')
    df = df.filter((day >= F.lit(start)) & (day < F.lit(end)))
    return write_parquet_partition(df, bronze_weather_directory, 'bronze_weather', snapshot_date_str)
