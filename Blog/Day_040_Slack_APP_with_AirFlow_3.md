# 39日目 Slack APP with AirFlow 2 Day38 Slack APP with AirFlow 2

本日の目標は
1. slack apiの使い方
2. webdriverの使い方
3. comic pusher slackの完成

## Step 1: slack apiの使い方
```python
# slack apiを通して、slackのchannelにメッセージを送る
send_notification = SlackAPIPostOperator(
    task_id='send_notification',
    # 事前にslack appを作成・権限設定・tokenを取得
    token=get_token(),
    # workspaceのどのchannel(app作成の際に、workspaceの指定は必要)
    channel='#random',
    text=get_message_text(),
    icon_url='http://airbnb.io/img/projects/airflow3.png'
)
```

## Step 2: webdriverの使い方
```python

def check_comic_info(**context):
    # get_read_historyのタスクから、metadataを取得
    metadata = context['task_instance'].xcom_pull(task_ids='get_read_history')
    # webdriverを定義
    driver = webdriver.Chrome()
    # crawlerの対象を定義
    driver.get('https://www.cartoonmad.com/')
    print("Arrived top page.")

    # jsonで格納している情報をall_comic_infoに格納
    all_comic_info = metadata
    # 更新フラグをfalseとする
    anything_new = False

    # 漫画を複数定義される場合、loopで取得する
    for comic_id, comic_info in dict(all_comic_info).items():

        comic_name = comic_info['name']
        print("Fetching {}'s chapter list..".format(comic_name))
        # comic_idは漫画サイトのid
        driver.get(comic_page_template.format(comic_id))

        # 最新連載の話数を取得
        links = driver.find_elements_by_partial_link_text('第')
        latest_chapter_num = [int(s) for s in links[-1].text.split() if s.isdigit()][0]
        previous_chapter_num = comic_info['previous_chapter_num']
        all_comic_info[comic_id]['latest_chapter_num'] = latest_chapter_num
        # 最新かどうかを確認
        all_comic_info[comic_id]['new_chapter_available'] = latest_chapter_num > previous_chapter_num
        if all_comic_info[comic_id]['new_chapter_available']:
            anything_new = True
            print("There are new chapter for {}(latest: {})".format(comic_name, latest_chapter_num))

    if not anything_new:
        print("Nothing new now, prepare to end the workflow.")

    # webdriverを閉じる
    driver.quit()

    return anything_new, all_comic_info
```


## Step 3: comic pusher slackの完成
```python
import os
import time
import json
import logging
from datetime import datetime, timedelta
from selenium import webdriver
from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.slack_operator import SlackAPIPostOperator
from airflow.operators.latest_only_operator import LatestOnlyOperator


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

# crawlの目標ページ
comic_page_template = 'https://www.cartoonmad.com/comic/{}.html'

# metadataの更新
def process_metadata(mode, **context):

    file_dir = os.path.dirname(__file__)
    # metadataをjsonに格納する
    metadata_path = os.path.join(file_dir, '../data/comic.json')
    # modeはreadの場合、jsonに保存してある情報を読む
    if mode == 'read':
        with open(metadata_path, 'r') as fp:
            metadata = json.load(fp)
            print("Read History loaded: {}".format(metadata))
            return metadata
    # modeはwriteの場合、jsonに保存してある情報を読む
    elif mode == 'write':
        print("Saving latest comic information..")
        # check_comic_infoのタスクから、all_comic_infoを取得
        _, all_comic_info = context['task_instance'].xcom_pull(task_ids='check_comic_info')

        # update to latest chapter
        for comic_id, comic_info in dict(all_comic_info).items():
            all_comic_info[comic_id]['previous_chapter_num'] = comic_info['latest_chapter_num']


        with open(metadata_path, 'w') as fp:
            json.dump(all_comic_info, fp, indent=2, ensure_ascii=False)


def check_comic_info(**context):
    # get_read_historyのタスクから、metadataを取得
    metadata = context['task_instance'].xcom_pull(task_ids='get_read_history')
    # webdriverを定義
    driver = webdriver.Chrome()
    driver.get('https://www.cartoonmad.com/')
    print("Arrived top page.")

    # jsonで格納している情報をall_comic_infoに格納
    all_comic_info = metadata
    # 更新フラグをfalseとする
    anything_new = False

    # 漫画を複数定義される場合、loopで取得する
    for comic_id, comic_info in dict(all_comic_info).items():

        comic_name = comic_info['name']
        print("Fetching {}'s chapter list..".format(comic_name))
        # comic_idは漫画サイトのid
        driver.get(comic_page_template.format(comic_id))

        # 最新連載の話数を取得
        links = driver.find_elements_by_partial_link_text('第')
        latest_chapter_num = [int(s) for s in links[-1].text.split() if s.isdigit()][0]
        previous_chapter_num = comic_info['previous_chapter_num']
        all_comic_info[comic_id]['latest_chapter_num'] = latest_chapter_num
        # 最新かどうかを確認
        all_comic_info[comic_id]['new_chapter_available'] = latest_chapter_num > previous_chapter_num
        if all_comic_info[comic_id]['new_chapter_available']:
            anything_new = True
            print("There are new chapter for {}(latest: {})".format(comic_name, latest_chapter_num))

    if not anything_new:
        print("Nothing new now, prepare to end the workflow.")

    # webdriverを閉じる
    driver.quit()

    return anything_new, all_comic_info


def decide_what_to_do(**context):
    anything_new, all_comic_info = context['task_instance'].xcom_pull(task_ids='check_comic_info')

    print("跟紀錄比較，有沒有新連載？")
    # new_chapter_availableはtrueの場合、pushを送るタスクを実行；異なる場合、dummyのタスクを実行
    if anything_new:
        return 'yes_generate_notification'
    else:
        return 'no_do_nothing'

# slack appのtokenを取得
def get_token():
    file_dir = os.path.dirname(__file__)
    token_path = os.path.join(file_dir, '../data/credentials/slack.json')
    with open(token_path, 'r') as fp:
        token = json.load(fp)['token']
        return token

def generate_message(**context):
    _, all_comic_info = context['task_instance'].xcom_pull(task_ids='check_comic_info')

    message = ''
    for comic_id, comic_info in all_comic_info.items():
        if comic_info['new_chapter_available']:
            name = comic_info['name']
            latest = comic_info['latest_chapter_num']
            prev = comic_info['previous_chapter_num']
            message += '{} Newest {} （Previous：{} ）\n'.format(name, latest, prev)
            message += comic_page_template.format(comic_id) + '\n\n'

    # slack appに送るメッセージをtxtに格納
    file_dir = os.path.dirname(__file__)
    message_path = os.path.join(file_dir, '../data/message.txt')
    with open(message_path, 'w') as fp:
        fp.write(message)

# slack appにmessage textを送る
def get_message_text():
    file_dir = os.path.dirname(__file__)
    message_path = os.path.join(file_dir, '../data/message.txt')
    with open(message_path, 'r') as fp:
        message = fp.read()

    return message


with DAG('comic_pusher_slack', default_args=default_args) as dag:

    # define tasks
    latest_only = LatestOnlyOperator(task_id='latest_only')

    get_read_history = PythonOperator(
        task_id='get_read_history',
        python_callable=process_metadata,
        op_args=['read'],
        provide_context=True
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

    # slack apiを通して、slackのchannelにメッセージを送る
    send_notification = SlackAPIPostOperator(
        task_id='send_notification',
        token=get_token(),
        channel='#random',
        text=get_message_text(),
        icon_url='http://airbnb.io/img/projects/airflow3.png'
    )

    do_nothing = DummyOperator(task_id='no_do_nothing')

    # define workflow
    latest_only >> get_read_history
    get_read_history >> check_comic_info >> decide_what_to_do
    decide_what_to_do >> generate_notification
    decide_what_to_do >> do_nothing
    generate_notification >> send_notification >> update_read_history

```

### 参考資料
airflow tutorial (Comic_app)  
https://leemeng.tw/a-story-about-airflow-and-data-engineering-using-how-to-use-python-to-catch-up-with-latest-comics-as-an-example.html
https://github.com/leemengtaiwan/gist-evernote#chrome-driver  
Slack app scope  
https://github.com/smallwins/slack/issues/67  
selenium  
https://qiita.com/yulily@github/items/c3edbe25e84280c17776  
https://kurozumi.github.io/selenium-python/locating-elements.html  
