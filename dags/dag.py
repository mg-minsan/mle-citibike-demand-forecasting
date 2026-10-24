from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run daily',
    schedule_interval='0 6 * * *',  
    start_date=datetime(2016, 1, 1),
    end_date=datetime(2017,12, 31),
    catchup=True,
) as dag:
    bronze_weather_table = BashOperator(
        task_id='bronze_weather_table',
        bash_command=('cd /app &&'
                      'PYTHONPATH=/app /opt/project-venv/bin/python scripts/bronze_weather_table.py  '
                      '--snapshotdate {{ ds }}'
                      )
    )

    silver_weather_table = BashOperator(
        task_id='silver_weather_table',
        bash_command=('cd /app && '
                      'PYTHONPATH=/app /opt/project-venv/bin/python scripts/silver_weather_table.py '
                      '--snapshotdate {{ ds }}'
                      )
    )

    bronze_trips = BashOperator(
        task_id='bronze_trips',
        bash_command=('cd /app && '
                      'PYTHONPATH=/app /opt/project-venv/bin/python scripts/bronze_trips.py '
                      '--snapshotdate {{ ds }}'
                      )
    )

    silver_trips = BashOperator(
        task_id='silver_trips',
        bash_command=('cd /app && '
                      'PYTHONPATH=/app /opt/project-venv/bin/python scripts/silver_trips.py '
                      '--snapshotdate {{ ds }}'
                      )
    )

    gold_label_store = BashOperator(
        task_id='gold_label_store',
        bash_command=('cd /app && '
                      'PYTHONPATH=/app /opt/project-venv/bin/python scripts/gold_label_store.py '
                      '--snapshotdate {{ ds }}'
                      )
    )

    gold_feature_store = BashOperator(
        task_id='gold_feature_store',
        bash_command=('cd /app && '
                      'PYTHONPATH=/app /opt/project-venv/bin/python scripts/gold_feature_store.py '
                      '--snapshotdate {{ ds }}'
                      )
    )

    bronze_weather_table >> silver_weather_table
    bronze_trips >> silver_trips
    silver_trips >> gold_label_store
    silver_trips >> gold_feature_store
    silver_weather_table >> gold_feature_store

