# 55日目 Prophet 2 Day55 Prophet 2

本日の目標は
1. 分析案1を実装（データ処理の練習）

## Step 1: 分析案1を実装（データ処理の練習）
```python
# 前年度の11月データを抽出
sales_201411 = sales[sales["date_block_num"] == 22]
sales_201411_sum = sales_201411["item_cnt_day"].sum()
sales_201411 = sales_201411.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"]
sales_201411["total_sales"] = sales_201411_sum

# 比率を計算する
sales_201411["percentage"] = sales["item_cnt_day"] / sales_201411["total_sales"]
print ("sales_201411")
print (sales_201411.head())

sales_201411["result"] = sales_201411["percentage"] * forecast["yhat"][0]
print ("calculation")
print (sales_201411.head())

submission = pd.merge(test, sales_201411, left_on=["shop_id", "item_id"], right_on=["shop_id", "item_id"])
print ("submission")
print (submission.head())

```


### 参考資料
Pandas  
http://pppurple.hatenablog.com/entry/2016/06/27/022310  
https://stackoverflow.com/questions/41815079/pandas-merge-join-two-data-frames-on-multiple-columns  
