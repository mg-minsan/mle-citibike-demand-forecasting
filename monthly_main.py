"""Build monthly partitions for trips, weather, labels and joined features."""
import argparse
import logging
import os
from pathlib import Path

from pyspark.sql import SparkSession

from utils_monthly import data_processing_bronze_table as bronze
from utils_monthly import data_processing_silver_table as silver
from utils_monthly import data_processing_gold_table as gold
from utils_monthly import data_processing_bronze_weather_table as bronze_weather
from utils_monthly import data_processing_silver_weather_table as silver_weather
from utils_monthly.utils import generate_snapshot_dates


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build monthly tables under datamart_monthly/')
    parser.add_argument('--startdate', default='2016-01-01', help='First month, YYYY-MM-DD')
    parser.add_argument('--enddate', default='2017-12-31', help='Last month, inclusive, YYYY-MM-DD')
    parser.add_argument('--layer', choices=('all', 'bronze', 'silver', 'gold'), default='all')
    args = parser.parse_args(argv)
    try:
        months = list(generate_snapshot_dates(args.startdate, args.enddate))
    except ValueError as exc:
        parser.error(str(exc))
    directories = {name: str(Path('datamart_monthly') / layer / table)
                   for name, layer, table in (
                       ('bronze_trips', 'bronze', 'trips'), ('silver_trips', 'silver', 'trips'),
                       ('bronze_weather', 'bronze', 'weather'), ('silver_weather', 'silver', 'weather'),
                       ('labels', 'gold', 'label_store'),
                       ('features', 'gold', 'feature_store'))}
    for directory in directories.values():
        Path(directory).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('PYSPARK_SUBMIT_ARGS', '--driver-memory 4g pyspark-shell')
    spark = (SparkSession.builder.appName('monthly_table_pipeline').master('local[2]')
             .config('spark.sql.shuffle.partitions', '8')
             .config('spark.sql.files.maxPartitionBytes', str(64 * 1024 * 1024))
             .config('spark.sql.adaptive.advisoryPartitionSizeInBytes', str(32 * 1024 * 1024))
             .config('spark.sql.legacy.timeParserPolicy', 'CORRECTED')
             .config('spark.sql.session.timeZone', 'America/New_York').getOrCreate())
    try:
        spark.sparkContext.setLogLevel('ERROR')
        for month in months:
            print('Processing month:', month[:7])
            if args.layer in ('all', 'bronze'):
                bronze.process_bronze_table(month, directories['bronze_trips'], spark)
                bronze_weather.process_bronze_weather_table(month, directories['bronze_weather'], spark)
            if args.layer in ('all', 'silver'):
                silver.process_silver_table(month, directories['bronze_trips'], directories['silver_trips'], spark)
                silver_weather.process_silver_weather_table(month, directories['bronze_weather'], directories['silver_weather'], spark)
            if args.layer in ('all', 'gold'):
                gold.process_labels_gold_table(month, directories['silver_trips'], directories['labels'], spark)
                gold.process_features_gold_table(month, directories['silver_trips'], directories['features'], spark,
                                                 silver_weather_directory=directories['silver_weather'])
    except BaseException:
        try:
            spark.stop()
        except Exception:
            logging.warning('Could not stop Spark after processing failed', exc_info=True)
        raise
    else:
        spark.stop()


if __name__ == '__main__':
    main()
