"""Shared same-day weather aggregation and feature definitions."""
import pyspark.sql.functions as F


WEATHER_FEATURE_COLUMNS = (
    'mean_temperature_c', 'med_temperature_c', 'max_temperature_c',
    'min_temperature_c', 'temperature_range_c', 'total_precipitation_mm',
    'total_rain_mm', 'mean_cloud_pct', 'mean_wind_speed_kmh',
    'max_wind_speed_kmh', 'wet_hours', 'hourly_rows',
    'temperature_hours', 'precipitation_hours',
)


def aggregate_daily_weather(df):
    """Summarize validated hourly observations by their local calendar date."""
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
    return agg_df
