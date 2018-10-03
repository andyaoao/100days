# 56日目 Prophet 3 Day56 Prophet 3

本日の目標は
1. 分析案2の仮説を立てる
2. 分析案2を実装（データ処理の練習）

## Step 1: 分析案2の仮説を立てる
分析案1のパフォーマンスが良くないため、分析案2を立てる。

仮説：売上の変化は点数に影響がない  
分析案：各商品の点数の月推移でモデルを構築、2015/11を予測。その後、2015/10の各店舗の点数配分をベースで、予測を配分する


## Step 2: 分析案2を実装（データ処理の練習）
```python
# 店舗別の売上データを整理する
monthly_shop_sales=sales.groupby(["date_block_num","shop_id"])["item_cnt_day"].sum()
# unstack関数で、行を列に変換(level 1 )
monthly_shop_sales=monthly_shop_sales.unstack(level=1)
monthly_shop_sales=monthly_shop_sales.fillna(0)
# datesをindexにする方法を調査中
# monthly_shop_sales.index=dates
# monthly_shop_sales=monthly_shop_sales.reset_index()
print ("monthly_shop_sales")
print (monthly_shop_sales.head())


# 上記で整理したデータをモデル構築する
start_time=time.time()

forecastsDict = {}
for node in range(len(monthly_shop_sales)):

    nodeToForecast = pd.concat([monthly_shop_sales.iloc[:,0], monthly_shop_sales.iloc[:, node+1]], axis = 1)
    print ("nodeToForecast")
    print (nodeToForecast.head())
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})
    growth = 'linear'
    m = Prophet(growth, yearly_seasonality=True)
    m.fit(nodeToForecast)
    future = m.make_future_dataframe(periods = 1, freq = 'MS')
    forecastsDict[node] = m.predict(future)

```


### 参考資料
Pandas  
https://note.nkmk.me/python-pandas-stack-unstack-pivot/  
https://note.nkmk.me/python-pandas-time-series-datetimeindex/  
