"""Calendar features from the supplied US federal holiday CSV."""
import csv
from datetime import date, timedelta
from pathlib import Path

import pyspark.sql.functions as F


DEFAULT_HOLIDAY_CSV = Path(__file__).resolve().parents[1] / 'data/us_federal_holidays_2016_2017.csv'
HOLIDAY_FEATURE_COLUMNS = ('fe_is_holiday', 'fe_is_observed_holiday')


def load_holiday_dates(holiday_csv=DEFAULT_HOLIDAY_CSV):
    """Return actual and observed dates separately, deduplicated for safe joins."""
    with Path(holiday_csv).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'date', 'observed_date'}.issubset(reader.fieldnames or []):
            raise ValueError('Holiday CSV must contain date and observed_date columns')
        rows = list(reader)
    if not rows:
        raise ValueError('Holiday CSV is empty')
    return ({date.fromisoformat(row['date']) for row in rows},
            {date.fromisoformat(row['observed_date']) for row in rows})


def long_weekend_dates(actual, observed):
    """Dates in 3+ consecutive weekend/holiday days, using only the supplied calendar.

    No bridging workday is assumed to be a day off. Extend the holiday CSV
    into adjacent years to capture breaks involving holidays beyond its range.
    """
    holidays = set(actual) | set(observed)
    if not holidays:
        return set()
    current = min(holidays) - timedelta(days=7)
    end = max(holidays) + timedelta(days=7)
    result, run = set(), []
    while current <= end:
        if current.weekday() >= 5 or current in holidays:
            run.append(current)
        else:
            if len(run) >= 3:
                result.update(run)
            run = []
        current += timedelta(days=1)
    if len(run) >= 3:
        result.update(run)
    return result


def add_holiday_features(df, holiday_csv=DEFAULT_HOLIDAY_CSV, include_long_weekend=False):
    """Flag actual and federally observed holiday dates without adding rows.

    Observed dates include holidays celebrated on their actual dates, and
    substitute weekdays where the supplied calendar lists one. This calendar
    covers 2016–2017; extend the source CSV when processing other years.
    """
    actual, observed = load_holiday_dates(holiday_csv)
    df = (df.withColumn('fe_is_holiday', F.col('snapshot_date').isin(sorted(actual)).cast('int'))
          .withColumn('fe_is_observed_holiday', F.col('snapshot_date').isin(sorted(observed)).cast('int')))
    if include_long_weekend:
        df = df.withColumn('fe_is_long_weekend', F.col('snapshot_date').isin(
            sorted(long_weekend_dates(actual, observed))).cast('int'))
    return df
