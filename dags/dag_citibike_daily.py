import pendulum
from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor

# daily prediction at 06:00 New York time, plus monthly retraining on the 1st of each month.
# Each run predicts for the day it fires on (data_interval_end), e.g. the run that fires 2016-07-01 06:00 predicts
# 2016-07-01. Tasks: wait for that day's gold feature partition -> retrain and explain (1st of the month only)
# -> predict -> export business CSV and monitor yesterday; optional SHAP runs separately after retraining.
# The repo is mounted at /app in the container; scripts run from /app so datamart/ paths resolve to /app/datamart.
APP = "/app"
# Use isolated project dependencies, not Airflow's own Python environment.
PYTHON = "/opt/project-venv/bin/python"
SNAPSHOT_DATE = "{{ data_interval_end.strftime('%Y-%m-%d') }}"
SNAPSHOT_FOLDER_DATE = "{{ data_interval_end.strftime('%Y_%m_%d') }}"
NY = "America/New_York"

with DAG(
    dag_id="dag-citibike-daily",
    description="daily station demand prediction, monthly model retraining",
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=1)},
    schedule_interval="0 6 * * *",
    start_date=pendulum.datetime(2016, 6, 30, tz=NY),   # first run predicts 2016-07-01 (first model is trained through 30 Jun)
    end_date=pendulum.datetime(2017, 10, 31, tz=NY),    # stops before Nov-Dec 2017, the untouched final holdout
    catchup=True,
    max_active_runs=1,
) as dag:

    wait_for_feature_partition = FileSensor(
        task_id="wait_for_feature_partition",
        filepath=APP + "/datamart/gold/feature_store/gold_feature_store_" + SNAPSHOT_FOLDER_DATE + ".parquet/_SUCCESS",
        fs_conn_id="fs_default",
        poke_interval=300,
        timeout=60 * 60 * 12,
        mode="reschedule",
    )

    retrain_model_monthly = BashOperator(
        task_id="retrain_model_monthly",
        bash_command="cd " + APP + " && if [ {{ data_interval_end.day }} -eq 1 ]; then "
                     "PYTHONPATH=" + APP + " " + PYTHON + " scripts/model_train.py --snapshotdate " + SNAPSHOT_DATE + "; "
                     "else echo 'not the 1st of the month, skipping retrain'; fi",
    )

    # explains the model the retrain just saved (SHAP on its test month) and checks attribution drift vs last month
    explain_model_monthly = BashOperator(
        task_id="explain_model_monthly",
        bash_command="cd " + APP + " && if [ {{ data_interval_end.day }} -eq 1 ] && [ \"${ENABLE_MONTHLY_SHAP:-1}\" = 1 ]; then "
                     "PYTHONPATH=" + APP + " " + PYTHON + " scripts/model_explain.py --snapshotdate " + SNAPSHOT_DATE + "; "
                     "else echo 'monthly SHAP disabled or not the 1st, skipping explanation'; fi",
    )

    predict_daily = BashOperator(
        task_id="predict_daily",
        bash_command="cd " + APP + " && PYTHONPATH=" + APP + " " + PYTHON + " scripts/model_predict_daily.py --snapshotdate " + SNAPSHOT_DATE,
    )

    export_predictions_csv = BashOperator(
        task_id="export_predictions_csv",
        bash_command="cd " + APP + " && PYTHONPATH=" + APP + " " + PYTHON + " scripts/export_predictions_csv.py --snapshotdate " + SNAPSHOT_DATE,
    )

    ensure_station_lookup = BashOperator(
        task_id="ensure_station_lookup",
        bash_command="cd " + APP + " && [ -f data/stations.csv ] || " + PYTHON + " scripts/extract_stations.py",
    )

    # checks yesterday (its real trip counts are known now): MAE vs baseline, 7-day alert, feature drift (PSI)
    monitor_daily = BashOperator(
        task_id="monitor_daily",
        bash_command="cd " + APP + " && PYTHONPATH=" + APP + " " + PYTHON + " scripts/model_monitor_daily.py --snapshotdate " + SNAPSHOT_DATE,
    )

    wait_for_feature_partition >> retrain_model_monthly >> predict_daily >> monitor_daily
    [predict_daily, ensure_station_lookup] >> export_predictions_csv
    retrain_model_monthly >> explain_model_monthly
