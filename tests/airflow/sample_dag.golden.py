from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta


default_args = {
    'owner': 'Datalab',
    'depends_on_past': False,
    'start_date': datetime(2015, 6, 13),
    'email': ['rajivpb@google.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG('print_dates', default_args=default_args)

print_utc_date = BashOperator(task_id='print_utc_date_id', bash_command='date -u', dag=dag)
print_pdt_date = BashOperator(task_id='print_pdt_date_id', bash_command='date', dag=dag)
print_utc_date.set_upstream(print_pdt_date)