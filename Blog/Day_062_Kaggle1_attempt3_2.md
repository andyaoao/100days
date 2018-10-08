# 61日目 Prophet 5 Day61 Prophet 5

本日の目標は
1. 分析案3の仮説を立てる
2. 分析案3を実装（データ処理の練習）

## Step 1: 分析案2の仮説を立てる
分析案1、2のパフォーマンスが良くないため、分析案3を立てる。

仮説：売上の変化は点数に影響がない  
分析案：店舗別商品別の点数の月推移でモデルを構築、2015/11を予測。


## Step 2: 分析案2を実装（データ処理の練習）
```python
# 日付データ型を整理
sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

# 商品別、店舗別、月別の集計
sales = sales.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].sum()
sales = sales.unstack().unstack()
sales = sales.fillna(0)
print ("sales")
print (sales.head())

dates = pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
sales.index=dates
sales = sales.reset_index()
print ("sales")
print (sales.head())

forecastsDict = {}
for node in range(len(sales)):

    nodeToForecast = pd.concat([sales.iloc[:,0], sales.iloc[:, node+1]], axis = 1)
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})
    if node < 1 :
        print ("nodeToForecast")
        print (nodeToForecast)
    growth = 'linear'
    m = Prophet(growth, yearly_seasonality=True)
    m.fit(nodeToForecast)
    future = m.make_future_dataframe(periods = 1, freq = 'MS')
    forecastsDict[node] = m.predict(future)

nCols = len(list(forecastsDict.keys()))+1
for key in range(0, nCols-1):
    f1 = np.array(forecastsDict[key].yhat)
    f2 = f1[:, np.newaxis]
    if key==0:
        predictions=f2.copy()
       # print(predictions.shape)
    else:
       predictions = np.concatenate((predictions, f2), axis = 1)

```


### 参考資料
Pandas  
https://note.nkmk.me/python-pandas-set-index/  
https://pythondatascience.plavox.info/pandas/%E8%A1%8C%E3%83%BB%E5%88%97%E3%81%AE%E9%95%B7%E3%81%95%E3%82%92%E7%A2%BA%E8%AA%8D  
https://note.nkmk.me/python-dict-keys-values-items/  
