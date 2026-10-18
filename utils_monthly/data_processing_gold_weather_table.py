import pyspark.sql.functions as F

from utils_monthly.utils import read_parquet_partition
from utils.weather_features import WEATHER_FEATURE_COLUMNS, aggregate_daily_weather, aggregate_weekly_weather
from utils.weather_validation import configure_weather_timezone
from utils_monthly.weather_validation import validate_monthly_weather



def build_daily_weather_features(snapshot_date_str, silver_weather_directory, spark):
    """Read one month of silver weather, then return daily gold aggregates.

    Used by the station/day feature builder without writing a separate
    gold weather partition.
    """
    configure_weather_timezone(spark)
    df = read_parquet_partition(silver_weather_directory, "silver_weather_", snapshot_date_str, spark)
    # Reject duplicates and incomplete measurements before they distort totals.
    validate_monthly_weather(df, snapshot_date_str)
    return aggregate_daily_weather(df)





def build_weekly_weather_features(snapshot_date_str, silver_weather_directory, spark):
    """Previous seven calendar days [D-7, D-1], including cross-month history.

    Aggregate available complete days. weather_days_7d exposes partial windows;
    with no history, coverage is zero and all measurement features stay null.
    """
    from datetime import timedelta
    from pathlib import Path
    from utils_monthly.utils import month_bounds, generate_snapshot_dates, monthly_partition_path

    start, end = month_bounds(snapshot_date_str)
    first = start - timedelta(days=7)
    history = None
    for month in generate_snapshot_dates(first.isoformat(), start.isoformat()):
        path = monthly_partition_path(silver_weather_directory, 'silver_weather', month)
        if not Path(path).exists():
            print(f'Missing weather history: {path}')
            continue
        daily = build_daily_weather_features(month, silver_weather_directory, spark)
        daily = daily.filter((F.col('snapshot_date') >= F.lit(first)) &
                             (F.col('snapshot_date') < F.lit(end - timedelta(days=1))))
        history = daily if history is None else history.unionByName(daily)
    dates = spark.range((end - start).days).select(
        F.date_add(F.lit(start), F.col('id').cast('int')).alias('snapshot_date'))
    return aggregate_weekly_weather(dates, history)
