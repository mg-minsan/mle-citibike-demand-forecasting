# Citi Bike Demand Forecasting

Forecast daily trip starts by station using NYC Citi Bike data from 2016–2017. PySpark builds bronze, silver, and gold tables; monthly training compares Poisson regression, random forest, and XGBoost against a lag-7 baseline. MLflow tracks models, while Airflow schedules predictions, retraining, and monitoring.

## Setup

Use Python 3.11 and a Java runtime compatible with PySpark 3.5.

```bash
python -m pip install -r requirements.txt
docker compose up -d --build airflow-init airflow-webserver airflow-scheduler
```

Open Airflow at http://localhost:8080 (`admin` / `admin`) and MLflow at http://localhost:5000. Rebuild the services after changing dependencies.

## Build the data

Place raw trip CSVs in `data/<year>-citibike-tripdata/<month>_*/*.csv` and weather data in `data/NYC_Weather_2016_2022.csv`. Holiday features use `data/us_federal_holidays_2016_2017.csv`.

```bash
python main.py --startdate 2016-01-01 --enddate 2017-12-31
```

Outputs go to `datamart/`. Process dates in order to supply the preceding 28 days of trip history. Use `--layer bronze|silver|gold` to rebuild one layer, or `main_table.py` to rebuild features only.

Weather features use same-day observations; day-ahead inference needs forecast weather or a model without weather inputs. Extend the holiday calendar for years beyond 2016–2017.

## Train, predict, and monitor

With MLflow running and gold history available, run from the repository root:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
PYTHONPATH=. python scripts/model_train.py --snapshotdate 2016-07-01
PYTHONPATH=. python scripts/model_predict_daily.py --snapshotdate 2016-07-01
PYTHONPATH=. python scripts/model_monitor_daily.py --snapshotdate 2016-07-02
```

Predictions default to date-aware historical model selection. Add `--model-selection live` to use the MLflow `champion` alias. Monitoring checks the previous day's predictions against actual demand, reporting performance, drift, and data-quality alerts. See [monitoring details](docs/model-monitoring.md).

The daily Airflow DAG schedules predictions at 06:00 New York time and retraining on the first of each month. Gold feature partitions must already exist.

## Project layout

- `utils/` — data processing, features, model storage, and monitoring.
- `scripts/` — training, prediction, monitoring, and reporting jobs.
- `dags/` — Airflow orchestration.
- `notebooks/` — exploration and model experiments.