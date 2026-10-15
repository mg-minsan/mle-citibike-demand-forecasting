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
    end_date=datetime(2017, 12, 31),
    catchup=True,
) as dag:
    bronze_weather_table = BashOperator(
        task_id='bronze_weather_table',
        bash_command=('cd /app &&'
                      'PYTHONPATH=/app python3 scripts/bronze_weather_table.py  '
                      '--snapshotdate {{ ds }}'
                      )
    )

    silver_weather_table = BashOperator(
        task_id='silver_weather_table',
        bash_command=('cd /app && '
                      'PYTHONPATH=/app python3 scripts/silver_weather_table.py '
                      '--snapshotdate {{ ds }}'
                      )
    )

    gold_weather_table = BashOperator(
        task_id='gold_weather_table',
        bash_command=('cd /app && '
                      'PYTHONPATH=/app python3 scripts/gold_weather_table.py '
                      '--snapshotdate {{ ds }}'
                      )
    )

    bronze_weather_table >> silver_weather_table >> gold_weather_table

