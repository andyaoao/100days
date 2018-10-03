# 56日目 Prophet 3 Day56 Prophet 3

本日の目標は
1. 分析案1を実装完成（データ処理の練習）

## Step 1: 分析案1を実装完成（データ処理の練習）
```python

# 日付データ型を整理
sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

# 前年度の11月データを抽出
sales_201411 = sales[sales["date_block_num"] == 22]
# as_index=Falseは実テーブルのように、一レコード一行で格納
sales_201411 = sales_201411.groupby(["date_block_num","shop_id","item_id"], as_index=False).sum()
print ("sales_201411")
print (sales_201411.head())

# 比率を計算する
# sum関数で全体のpercentageを算出
sales_201411["percentage"] = sales_201411["item_cnt_day"] / sales_201411["item_cnt_day"].sum()
print ("sales_201411 percentage")
print (sales_201411.head())

# date block num(年月)をベースで全社の販売点数を積上げる
ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
print ("ts")
print (ts.head())

# prophetが受け入れるデータ形は、日付(ds)と値(y)
ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
ts=ts.reset_index()
# 列名を修正する
ts.columns=['ds','y']
print ("before modeling")
print (ts.head())

#時系列モデルを定義
# パラメータは、年周期があること
model = Prophet('linear', yearly_seasonality=True)
model.fit(ts)

# 2017/11を予測
future = model.make_future_dataframe(periods = 5, freq = 'MS')
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

print ("forecast")
print (forecast.head())

# 5期を予測したが、2018/11はその第1期
sales_201411["result"] = sales_201411["percentage"] * forecast["yhat"][0]
print ("calculation")
print (sales_201411.head())

submission = pd.merge(test, sales_201411, left_on=["shop_id", "item_id"], right_on=["shop_id", "item_id"], how='left')
print ("submission before delete")
print (submission.head())
submission_new = submission.drop(['date_block_num', 'shop_id', 'item_id','item_price','item_cnt_day', 'percentage'], axis=1)
submission_new = submission_new.fillna(0)
print ("submission after delete")
print (submission_new.head())

submission_new.columns=['ID','item_cnt_month']
print ("submission")
print (submission_new.head())

# csvに書き出し
submission_new.to_csv("./Datasets/PredictFutureSales/submission.csv", index=False)


```


### 参考資料
Pandas  
https://blog.csdn.net/zhili8866/article/details/68134481  
https://blog.csdn.net/claroja/article/details/65661826  
http://sinhrks.hatenablog.com/entry/2015/01/28/073327  
https://note.nkmk.me/python-pandas-to-csv/  
https://qiita.com/stokes/items/945115525ca36a3dcf7c  
https://qiita.com/propella/items/a9a32b878c77222630ae#%E5%8F%82%E8%80%83  
https://stackoverflow.com/questions/24980437/pandas-groupby-and-then-merge-on-original-table  
