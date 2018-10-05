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
# 前月のデータを抽出
sales_201510 = sales[sales["date_block_num"] == 33]
# as_index=Falseは実テーブルのように、一レコード一行で格納
# 2015/10の商品別、店舗別の集計
sales_201510 = sales_201510.groupby(["shop_id", "item_id"])["item_cnt_day"].sum()

print ("sales_201510")
print (sales_201510.head())

sales_201510 = sales_201510.unstack()
sales_201510 = sales_201510.fillna(0)
print ("sales_201510_shop")
print (sales_201510.head())

# 比率を計算する
# applyを使用して、列内を計算
sales_201510 = sales_201510.apply(lambda x: x/sum(x))
print ("sales_201510_percentage")
print (sales_201510.head())

sales_201510 = sales_201510.stack()
sales_201510 = sales_201510.reset_index()
print ("sales_201510_big_table")
print (sales_201510.head())

```


### 参考資料
Pandas  
https://note.nkmk.me/python-pandas-set-index/  
https://pythondatascience.plavox.info/pandas/%E8%A1%8C%E3%83%BB%E5%88%97%E3%81%AE%E9%95%B7%E3%81%95%E3%82%92%E7%A2%BA%E8%AA%8D  
https://note.nkmk.me/python-dict-keys-values-items/  
