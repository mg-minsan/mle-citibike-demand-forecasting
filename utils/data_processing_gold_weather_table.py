from utils.utils import write_parquet_partition, read_parquet_partition
from utils.weather_features import aggregate_daily_weather
from utils.weather_validation import configure_weather_timezone, validate_hourly_weather



def build_daily_weather_features(snapshot_date_str, silver_weather_directory, spark):
    """Read and validate silver hourly weather, then return daily gold aggregates.

    Shared by the standalone weather writer and station/day feature builder;
    this function does not write a separate gold weather partition.
    """
    timezone = configure_weather_timezone(spark)
    df = read_parquet_partition(silver_weather_directory, "silver_weather_", snapshot_date_str, spark)
    # Reject duplicates and incomplete measurements before they distort totals.
    validate_hourly_weather(df, snapshot_date_str, timezone)
    return aggregate_daily_weather(df)


def process_gold_weather_table(snapshot_date_str, silver_weather_directory, gold_weather_directory, spark):
    agg_df = build_daily_weather_features(snapshot_date_str, silver_weather_directory, spark)
    return write_parquet_partition(agg_df, gold_weather_directory, "gold_weather_", snapshot_date_str)
