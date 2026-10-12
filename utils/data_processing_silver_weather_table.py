import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, TimestampType
from utils.utils import read_csv_partition, write_parquet_partition
from utils.weather_validation import configure_weather_timezone, validate_hourly_weather


def process_silver_weather_table(snapshot_date_str, bronze_weather_directory,  silver_weather_directory,  spark):
    timezone = configure_weather_timezone(spark)
    df = read_csv_partition(
        bronze_weather_directory, "bronze_weather_", snapshot_date_str, spark,
        infer_schema=False,
    )

    column_name_map = {
        "time": "time",
        "temperature_2m (°C)": "temperature_c",
        "precipitation (mm)": "precipitation_mm",
        "rain (mm)": "rain_mm",
        "cloudcover (%)": "cloud_pct",
        "cloudcover_low (%)": "cloud_low_pct",
        "cloudcover_mid (%)": "cloud_mid_pct",
        "cloudcover_high (%)": "cloud_high_pct",
        "windspeed_10m (km/h)": "wind_speed_kmh",
        "winddirection_10m (°)": "wind_direction_deg",
    }
    missing_columns = sorted(set(column_name_map) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing source weather columns: {missing_columns}")
    for original_name, new_name in column_name_map.items():
        df = df.withColumnRenamed(original_name, new_name)

    column_type_map = {
        "time": TimestampType(),
        "temperature_c": DoubleType(),
        "precipitation_mm": DoubleType(),
        "rain_mm": DoubleType(),
        "cloud_pct": DoubleType(),
        "cloud_low_pct": DoubleType(),
        "cloud_mid_pct": DoubleType(),
        "cloud_high_pct": DoubleType(),
        "wind_speed_kmh": DoubleType(),
        "wind_direction_deg": DoubleType(),
    }

    # Cast the renamed columns; doubles avoid truncating measurements.
    for column, data_type in column_type_map.items():
        df = df.withColumn(column, F.col(column).cast(data_type))

    validate_hourly_weather(df, snapshot_date_str, timezone)
    return write_parquet_partition(df, silver_weather_directory, "silver_weather_", snapshot_date_str)