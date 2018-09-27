# 50日目 Time Series Data processing 4 Day48 Time Series Data processing 4

本日の目標は
1. システム方法で最適ARMAを探す
2. ARMAモデルで予測

## Step 1: 定常過程の時系列モデルの比較

```python
# ARMAモデルパラメータの最適化
# AIC(Akaike Infomation criterion)指標
best_aic = np.inf
best_order = None
best_mdl = None

# ARとMAのパラメータを5まで設定
rng = range(5)
for i in rng:
    for j in rng:
        try:
            # arma22で作成したモデルなので、arma22は最適のはず
            tmp_mdl = smt.ARMA(arma22, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            # ベストaic(最小値)を選出
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue

print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
# ベストaicはARMA(2,2)

best_aic = np.inf
best_order = None
best_mdl = None

rng = range(5)
for i in rng:
    for j in rng:
        try:
            # 売り上げデータを最適ARMAモデルを探す
            tmp_mdl = smt.ARMA(new_ts.values, order=(i, j)).fit(method='mle', trend='nc')
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i, j)
                best_mdl = tmp_mdl
        except: continue


print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
# ベストaicはARMA(1,1)
```

## Step 2: ARMAモデルで予測
```python
# 予測
ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
ts=ts.reset_index()
```


### 参考資料
AIC  
http://hclab.sakura.ne.jp/stress_nervous_ar_aic.html  
https://blog.csdn.net/qq_37267015/article/details/71410480  
