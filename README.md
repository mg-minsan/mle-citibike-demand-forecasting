# Citi Bike Demand Forecasting

Forecast daily trip starts by station using NYC Citi Bike data from 2016–2017. PySpark builds bronze, silver, and gold tables; monthly training compares Poisson regression, random forest, and XGBoost against a lag-7 baseline. MLflow tracks models, while Airflow schedules predictions, retraining, and monitoring.

## Setup

Use Python 3.11 and a Java runtime compatible with PySpark 3.5.

```bash
python -m pip install -r requirements.txt
docker compose up -d --build airflow-init airflow-webserver airflow-scheduler streamlit
```

Open Airflow at http://localhost:8080 (`admin` / `admin`), MLflow at http://localhost:5000, and the Streamlit dashboard at http://localhost:8501. Rebuild the services after changing dependencies.

## Build the data

Place raw trip CSVs in `data/<year>-citibike-tripdata/<month>_*/*.csv` and weather data in `data/NYC_Weather_2016_2022.csv`. Holiday features use `data/us_federal_holidays_2016_2017.csv`.

```bash
python main.py --startdate 2016-01-01 --enddate 2017-12-31
```

Outputs go to `datamart/`. Process dates in order to supply the preceding 28 days of trip history. Use `--layer bronze|silver|gold` to rebuild one layer, or `main_table.py` to rebuild features only.

Weather features use same-day observations; day-ahead inference needs forecast weather or a model without weather inputs. Extend the holiday calendar for years beyond 2016–2017.

## Create a small trip sample

To create a reproducible random sample of **50 trips per day**, run:

```bash
python scripts/sample_trips.py
```

The script streams the raw CSVs in `data/` without loading the full dataset into memory.
It leaves the originals intact and preserves the original trip CSV filenames and
relative folders under `data_sample/`. For example,
`data/2016-citibike-tripdata/2_February/201602-citibike-tripdata_1.csv`
becomes `data_sample/2016-citibike-tripdata/2_February/201602-citibike-tripdata_1.csv`.
Each selected trip stays in its original source file; the limit is 50 trips per day
**across all files**, not 50 per file. The script also writes
`data_sample/sample_summary.csv` with original and sampled counts for each date.
Days with fewer than 50 trips retain all available trips; dates absent from the source
are not created. Jersey City files are excluded. All original columns are retained.

Use `--input-dir`, `--output-dir`, `--trips-per-day`, or `--seed` to customize the run.
The default seed is `42`. Existing samples are overwritten by default. To regenerate, run:

```bash
python scripts/sample_trips.py
```

This replaces sample trip CSVs and summary files, removing obsolete sample CSVs.
Original files in `data/` are unchanged.
Use `--no-overwrite` to refuse an existing output folder instead.
The sample is suitable for submission or demonstrations, but its counts do not
represent full station demand. The main pipeline still reads `data/` by default.

## Station lookup for dashboards

Run `python scripts/extract_stations.py` to generate `data/stations.csv` with
`station_id`, `station_name`, `latitude`, and `longitude`. It reads both start and
end stations from the full NYC trip dataset and writes one row per station ID.
When an ID has multiple names or locations, it uses the most frequently observed
name/coordinate combination. This is a historical lookup, including operational
stations present in the source, rather than a list of currently active stations.
Read `station_id` as a string when joining it to predictions.

## Train, predict, and monitor

With MLflow running and gold history available, run from the repository root:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
PYTHONPATH=. python scripts/model_train.py --snapshotdate 2016-07-01
PYTHONPATH=. python scripts/model_predict_daily.py --snapshotdate 2016-07-01
python scripts/export_predictions_csv.py --snapshotdate 2016-07-01
PYTHONPATH=. python scripts/model_monitor_daily.py --snapshotdate 2016-07-02
```

Predictions default to date-aware historical model selection. Add `--model-selection live` to use the MLflow `champion` alias. Monitoring checks the previous day's predictions against actual demand, reporting performance, drift, and data-quality alerts. See [monitoring details](docs/model-monitoring.md).

The daily Airflow DAG schedules predictions at 06:00 New York time and retraining on the first of each month. Gold feature partitions must already exist.

After each successful prediction task, Airflow exports a business CSV to
`datamart/business/predictions/predictions_YYYY_MM_DD.csv`. It contains exactly
`station_id`, `station_name`, `model_prediction`, and `date`, in that order, with one row per station.
`date` is the forecast date; `model_prediction` is the predicted trip-start count
(rounded to the nearest whole trip, with .5 rounded up). The source Parquet keeps
the original prediction precision. Names come from `data/stations.csv`, which must be
generated before exporting. Missing names fail the export rather than dropping
stations. Reruns replace that day's CSV. The standalone export command above can
also export existing prediction partitions without rerunning the model.

After CSV export succeeds, `email_predictions` runs as an Airflow
`EmptyOperator` placeholder for future email delivery. It does not send email.

## Streamlit prediction dashboard

The dashboard (`streamlit/app.py`) displays the business CSVs named
`predictions_YYYY_MM_DD.csv` in `datamart/business/predictions/`. It does not need
MLflow, Spark, or Airflow to be running. Export at least one day first (see above).

Run it with Docker Compose:

```bash
docker compose up -d --build streamlit
```

Compose mounts `./datamart` read-only at `/app/datamart`, so new exports show up
without a rebuild. The app code is copied into the image, so rebuild the service
after changing `streamlit/`.

Or run it locally (run from `streamlit/` so `.streamlit/config.toml` applies the theme):

```bash
python -m pip install -r streamlit/requirements.txt
cd streamlit
python -m streamlit run app.py
```

Open http://localhost:8501. The sidebar has a forecast date picker (defaults to
the latest file), a station filter (search by name or ID; empty means all
stations), and a **Refresh forecasts** button to reload files from disk. The page shows:

- total predicted trip starts, station count, and average per station for the selection;
- a bar chart of the top 15 stations and a spotlight on the busiest station with its share of demand;
- the prediction table, with a download of the selected rows using the four business columns.

A file is rejected with an error if it is missing a column, has blank or duplicate
station IDs, has dates that differ from the date in its filename, or has negative
or non-numeric predictions. Set `PREDICTIONS_CSV_DIR` to read another folder;
relative paths resolve from the repository root, not the current directory.

## Project layout

- `streamlit/` — prediction dashboard app, Dockerfile, and dashboard dependencies.
- `utils/` — data processing, features, model storage, and monitoring.
- `scripts/` — training, prediction, monitoring, and reporting jobs.
- `dags/` — Airflow orchestration.
- `notebooks/` — exploration and model experiments.
