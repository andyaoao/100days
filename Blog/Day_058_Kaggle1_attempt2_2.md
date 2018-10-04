# 58日目 Prophet 3 Day58 Prophet 3

本日の目標は
1. 分析案2の仮説を立てる
2. 分析案2を実装（データ処理の練習）

## Step 1: 分析案2の仮説を立てる
分析案1のパフォーマンスが良くないため、分析案2を立てる。

仮説：売上の変化は点数に影響がない  
分析案：各商品の点数の月推移でモデルを構築、2015/11を予測。その後、2015/10の各店舗の点数配分をベースで、予測を配分する


## Step 2: 分析案2を実装（データ処理の練習）
```python
# 店舗ごとに時系列データを作る
forecastsDict = {}
for node in range(len(monthly_shop_sales)):

    nodeToForecast = pd.concat([monthly_shop_sales.iloc[:,0], monthly_shop_sales.iloc[:, node+1]], axis = 1)
    # print ("nodeToForecast")
    # print (nodeToForecast.head())
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})
    growth = 'linear'
    # mを投入したのは、20130101-20151001の期間
    m = Prophet(growth, yearly_seasonality=True)
    m.fit(nodeToForecast)
    # futureの出力は、20130101-20151101の期間(periodは1のため、元の系列に1期間追加)
    future = m.make_future_dataframe(periods = 1, freq = 'MS')
    forecastsDict[node] = m.predict(future)

# 全店舗の予測を一つのdfに整理
predictions = np.zeros([len(forecastsDict[0].yhat),1])
print ("predictions")
print (predictions)

# 全てのcolumn数は、
nCols = len(list(forecastsDict.keys()))+1
print ("nCols")
print (nCols)

for key in range(0, nCols-1):
    f1 = np.array(forecastsDict[key].yhat)
    f2 = f1[:, np.newaxis]
    if key==0:
        predictions=f2.copy()
       # print(predictions.shape)
    else:
       predictions = np.concatenate((predictions, f2), axis = 1)

print ("prediction after")
print (predictions)

```


### 参考資料
Pandas  
https://note.nkmk.me/python-pandas-set-index/  
https://pythondatascience.plavox.info/pandas/%E8%A1%8C%E3%83%BB%E5%88%97%E3%81%AE%E9%95%B7%E3%81%95%E3%82%92%E7%A2%BA%E8%AA%8D  
https://note.nkmk.me/python-dict-keys-values-items/  
