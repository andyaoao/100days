import time
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# ジョブの実行パラメター
default_args = {
    'owner': 'Box',
    # ジョブの開始時間
    'start_date': datetime(2018, 9, 12, 0, 0),
    # ジョブの回す頻度
    'schedule_interval': '@daily',
    # 失敗した時、何回リトライするか
    'retries': 2,
    # リトライのdelay時間
    'retry_delay': timedelta(minutes=1)
}

# ジョブで処理の内容
def fn_time():
    print("time:", time.time())

# ジョブをDAGに登録する
with DAG('comic_app_v1', default_args=default_args) as dag:
    time = PythonOperator(
        task_id='time',
        python_callable=fn_time
    )
