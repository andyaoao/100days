# 39日目 Slack APP with AirFlow 2 Day38 Slack APP with AirFlow 2

本日の目標は
1. 環境構築
2. taskの関係性

## Step 1: 環境構築
1. seleniumをインストール  
conda install -c conda-forge selenium  
2. Chrome Driverをダウンロード
3. Chrome Driverを/usr/local/binに格納する
4. Slack appを作成
5. Slack appのtokenを取得する
6. Slack appにchat:write:botの権限を設定

## Step 2: taskの関係性
```python

# 一つ目のタスク：metadataの更新
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
        # 上流のタスクの結果(return値)を使う
        _, all_comic_info = context['task_instance'].xcom_pull(task_ids='check_comic_info')

        # 最新状態をjsonに更新
        for comic_id, comic_info in dict(all_comic_info).items():
            all_comic_info[comic_id]['previous_chapter_num'] = comic_info['latest_chapter_num']

        with open(metadata_path, 'w') as fp:
            json.dump(all_comic_info, fp, indent=2, ensure_ascii=False)

# ２個目のタスク：最新連載の確認
def check_comic_info(**context):
    metadata = context['task_instance'].xcom_pull(task_ids='get_read_history')
    driver = webdriver.Chrome()
    driver.get('https://www.cartoonmad.com/')
    print("Arrived top page.")

    # ・・・

    return anything_new, all_comic_info
```


### 参考資料
airflow tutorial (Comic_app)  
https://leemeng.tw/a-story-about-airflow-and-data-engineering-using-how-to-use-python-to-catch-up-with-latest-comics-as-an-example.html
https://github.com/leemengtaiwan/gist-evernote#chrome-driver  
Slack app scope  
https://github.com/smallwins/slack/issues/67  
