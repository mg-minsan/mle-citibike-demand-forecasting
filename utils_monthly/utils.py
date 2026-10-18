"""Month boundaries and monthly Parquet partitions."""
from datetime import datetime, timedelta
from pathlib import Path


def month_bounds(snapshot_date_str):
    start = datetime.strptime(snapshot_date_str, '%Y-%m-%d').date().replace(day=1)
    end = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return start, end


def generate_snapshot_dates(start_date_str='2016-01-01', end_date_str='2017-12-31'):
    if datetime.strptime(start_date_str, '%Y-%m-%d') > datetime.strptime(end_date_str, '%Y-%m-%d'):
        raise ValueError('Start date must be on or before end date.')
    current, _ = month_bounds(start_date_str)
    last, _ = month_bounds(end_date_str)
    while current <= last:
        yield current.isoformat()
        _, current = month_bounds(current.isoformat())


def monthly_partition_path(directory, prefix, snapshot_date_str):
    start, _ = month_bounds(snapshot_date_str)
    return str(Path(directory) / f'{prefix.rstrip("_")}_{start:%Y_%m}.parquet')


def read_parquet_partition(directory, prefix, snapshot_date_str, spark):
    return spark.read.parquet(monthly_partition_path(directory, prefix, snapshot_date_str))


def write_parquet_partition(df, directory, prefix, snapshot_date_str):
    filepath = monthly_partition_path(directory, prefix, snapshot_date_str)
    df.write.mode('overwrite').parquet(filepath)
    print('saved to:', filepath)
    return df
