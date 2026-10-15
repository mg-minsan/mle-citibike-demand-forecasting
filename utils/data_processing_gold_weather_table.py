import pyspark.sql.functions as F

from utils.utils import write_parquet_partition, read_parquet_partition
from utils.weather_validation import configure_weather_timezone, validate_hourly_weather



def process_gold_weather_table(snapshot_date_str, silver_weather_directory,  gold_weather_directory,  spark):
    timezone = configure_weather_timezone(spark)
    df = read_parquet_partition(silver_weather_directory, "silver_weather_", snapshot_date_str, spark)
    # Reject duplicates and incomplete measurements before they distort totals.
    validate_hourly_weather(df, snapshot_date_str, timezone)
    agg_df = (
        df 
        .groupBy(F.to_date("time").alias("snapshot_date"))
        .agg(
            F.avg("temperature_c").alias("mean_temperature_c"),
            F.median("temperature_c").alias("med_temperature_c"),
            F.max("temperature_c").alias("max_temperature_c"),
            F.min("temperature_c").alias("min_temperature_c"),
            (F.max("temperature_c") - F.min("temperature_c")).alias("temperature_range_c"),

            F.sum("precipitation_mm").alias("total_precipitation_mm"),
            F.sum("rain_mm").alias("total_rain_mm"),

            F.avg("cloud_pct").alias("mean_cloud_pct"),

            F.avg("wind_speed_kmh").alias("mean_wind_speed_kmh"),
            F.max("wind_speed_kmh").alias("max_wind_speed_kmh"),

            F.sum(
                F.when(F.col("precipitation_mm") > 0, 1).otherwise(0)
            ).alias("wet_hours"),

            # Coverage checks
            F.count("*").alias("hourly_rows"),
            F.count("temperature_c").alias("temperature_hours"),
            F.count("precipitation_mm").alias("precipitation_hours"),
        )
        .orderBy("snapshot_date")
    )
    return write_parquet_partition(agg_df, gold_weather_directory, "gold_weather_", snapshot_date_str)

