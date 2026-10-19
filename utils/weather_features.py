"""Shared daily aggregation and past-seven-day weather feature definitions."""
import pyspark.sql.functions as F


WEATHER_FEATURE_COLUMNS = (
    'mean_temperature_c_7d', 'min_temperature_c_7d', 'max_temperature_c_7d',
    'temperature_range_c_7d', 'total_precipitation_mm_7d', 'wet_days_7d',
    'wet_hours_7d', 'mean_cloud_pct_7d', 'mean_wind_speed_kmh_7d',
    'max_wind_speed_kmh_7d', 'weather_days_7d',
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



def aggregate_weekly_weather(dates, history):
    """Aggregate [D-7, D-1]; partial windows expose their day count."""
    if history is None:
        for name in WEATHER_FEATURE_COLUMNS:
            dates = dates.withColumn(name, F.lit(0).cast('int') if name == 'weather_days_7d'
                                     else F.lit(None).cast('double'))
        return dates

    dates, history = dates.alias('dates'), history.alias('history')
    joined = dates.join(history,
                        (F.col('history.snapshot_date') >= F.date_sub(F.col('dates.snapshot_date'), 7)) &
                        (F.col('history.snapshot_date') < F.col('dates.snapshot_date')), 'left')
    return joined.groupBy(F.col('dates.snapshot_date').alias('snapshot_date')).agg(
        F.avg('mean_temperature_c').alias('mean_temperature_c_7d'),
        F.min('min_temperature_c').alias('min_temperature_c_7d'),
        F.max('max_temperature_c').alias('max_temperature_c_7d'),
        (F.max('max_temperature_c') - F.min('min_temperature_c')).alias('temperature_range_c_7d'),
        F.sum('total_precipitation_mm').alias('total_precipitation_mm_7d'),
        F.sum(F.when(F.col('history.snapshot_date').isNotNull(),
                     (F.col('total_precipitation_mm') > 0).cast('int'))).cast('double').alias('wet_days_7d'),
        F.sum('wet_hours').cast('double').alias('wet_hours_7d'),
        F.avg('mean_cloud_pct').alias('mean_cloud_pct_7d'),
        F.avg('mean_wind_speed_kmh').alias('mean_wind_speed_kmh_7d'),
        F.max('max_wind_speed_kmh').alias('max_wind_speed_kmh_7d'),
        F.count('history.snapshot_date').cast('int').alias('weather_days_7d'),
    )
