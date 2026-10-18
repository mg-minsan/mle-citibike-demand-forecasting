"""Validate a complete month of hourly local-clock silver measurements."""
import pyspark.sql.functions as F

from utils.weather_validation import WEATHER_MEASUREMENTS
from utils_monthly.utils import month_bounds


def validate_monthly_weather(df, snapshot_date_str):
    start, end = month_bounds(snapshot_date_str)
    missing = sorted({'time', *WEATHER_MEASUREMENTS} - set(df.columns))
    if missing:
        raise ValueError(f'Missing weather columns: {missing}')
    invalid = F.col('time').isNull() | ~F.col('time').cast('string').rlike(
        r'^\d{4}-\d{2}-\d{2} \d{2}:00:00$')
    for name in WEATHER_MEASUREMENTS:
        value = F.col(name)
        invalid = invalid | value.isNull() | F.isnan(value) | (F.abs(value) == float('inf'))
    # Silver timestamps are local clock labels (timestamp_ntz), just like daily silver.
    days = df.groupBy(F.to_date('time').alias('day')).agg(
        F.count('*').alias('hours'), F.countDistinct('time').alias('distinct_hours'),
        F.count(F.when(invalid, 1)).alias('invalid_rows'),
    ).collect()
    if (len(days) != (end - start).days or
            any(r.day is None or not start <= r.day < end or r.hours != 24 or
                r.distinct_hours != 24 or r.invalid_rows for r in days)):
        raise ValueError(f'Expected 24 valid, distinct local-clock hours per day throughout {start:%Y-%m}')
