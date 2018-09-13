# 37日目 AirFlowの理解 Day36 Understanding AirFlow 2

本日の目標は
1. Airflowのジョブ定義
2. ジョブ内のタスク定義
3. タスクの実行順番、条件定義

## Step 1: Airflowのジョブ定義
```python
import time
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.slack_operator import SlackAPIPostOperator

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
```

## Step 2: ジョブ内のタスク定義

```python
# 各タスクの定義（漫画連載のプッシュ通知）
# 履歴を読み込む（書き込む）
def process_metadata(mode, **context):
    if mode == 'read':
        print("read history")
    elif mode == 'write':
        print("update history")

# 新のchapterが出るかどうかのチェック
def check_comic_info(**context):
    all_comic_info = context['task_instance'].xcom_pull(task_ids='get_read_history')
    print("check for the new chapters")

    anything_new = time.time() % 2 > 1
    return anything_new, all_comic_info

# 新のchapterが出るかどうかをベースで次のアクションを判断する
def decide_what_to_do(**context):
    anything_new, all_comic_info = context['task_instance'].xcom_pull(task_ids='check_comic_info')

    print("compare to the history, check if new one is up or not")
    if anything_new:
        # 新のchapterが出ているなら、メッセージを送るタスクを実行
        return 'yes_generate_notification'
    else:
        # 新のchapterが出ていないなら、dummyのタスクを実行
        return 'no_do_nothing'

# 新のchapterが出るかどうかをベースで次のアクションを判断する
def generate_message(**context):
    _, all_comic_info = context['task_instance'].xcom_pull(task_ids='check_comic_info')
    print("export the message to a file")
```

## Step 3: タスクの実行順番、条件定義

```python
with DAG('comic_pusher', default_args=default_args) as dag:

    get_read_history = PythonOperator(
        task_id='get_read_history',
        python_callable=process_metadata,
        op_args=['read']
    )

    check_comic_info = PythonOperator(
        task_id='check_comic_info',
        python_callable=check_comic_info,
        provide_context=True
    )

    decide_what_to_do = BranchPythonOperator(
        task_id='new_comic_available',
        python_callable=decide_what_to_do,
        provide_context=True
    )

    update_read_history = PythonOperator(
        task_id='update_read_history',
        python_callable=process_metadata,
        op_args=['write'],
        provide_context=True
    )

    generate_notification = PythonOperator(
        task_id='yes_generate_notification',
        python_callable=generate_message,
        provide_context=True
    )

    send_notification = SlackAPIPostOperator(
        task_id='send_notification',
        token="YOUR_SLACK_TOKEN",
        channel='#comic-notification',
        text="[{{ ds }}] New chapter is released",
        icon_url='http://airbnb.io/img/projects/airflow3.png'
    )

    do_nothing = DummyOperator(task_id='no_do_nothing')

    # ワークフローの定義
    get_read_history >> check_comic_info >> decide_what_to_do

    decide_what_to_do >> generate_notification
    decide_what_to_do >> do_nothing

    generate_notification >> send_notification >> update_read_history

```
### 参考資料
airflow tutorial (Comic_app)
https://leemeng.tw/a-story-about-airflow-and-data-engineering-using-how-to-use-python-to-catch-up-with-latest-comics-as-an-example.html
