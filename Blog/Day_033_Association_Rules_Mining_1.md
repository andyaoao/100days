# 33日目 相関ルール 1 Day33 Association Rules Mining 1

本日の目標は
1. 相関ルールのロジック理解
2. 相関ルールの実装(前半)

## Step 1: 相関ルールのロジック理解
相関ルールの目的は、傾向をわかることだ。A商品を買った客は、B商品を買う確率を把握し、より確実な推薦を作る。  
support(支持率)：特定の商品は全体の占め率。  
confidence：A商品を買った客は、B商品を買う確率。支持率より算出する。  
support(a,b) / support(a)  
lift：confidenceで算出の結果はA->Bなので、A->BとB->A両方の結果を考慮した上の値だ(coincidenceを回避するため)。  
support(a,b) / support(a) * support(b)  

## Step 2: 相関ルールの実装(前半)
```python
import pandas as pd
import numpy as np
import sys
from itertools import combinations, groupby
from collections import Counter

orders = pd.read_csv('./Datasets/AssociateRulesSample.csv')
print (orders)

# orderのdfをseriesに変える
orders = orders.set_index('Order_ID')['Item']
print (orders)

# Returns frequency counts for items and item pairs
def freq(iterable):
    if type(iterable) == pd.core.series.Series:
        return iterable.value_counts().rename("freq")
    else:
        return pd.Series(Counter(iterable)).rename("freq")

# orderの数(レシート数)を計算
def order_count(order_item):
    return len(set(order_item.index))

def association_rules(order_item, min_support):

    print("Starting order_item: {:22d}".format(len(order_item)))


    # 各itemの出現回数を計算(支持率)
    # order_count:レシートの数
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # 設定された最小単位の支持率を超えているitemは対象となる（対象となるitemのindexを抽出し、order_item(分析対象)に入れる）
    qualifying_items       = item_stats[item_stats['support'] >= min_support].index
    order_item             = order_item[order_item.isin(qualifying_items)]

    print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))
    print("Remaining order_item: {:21d}".format(len(order_item)))


    # 2商品以上買っているレシートは分析の対象となる（相関ルールが作れるため）
    order_size             = freq(order_item.index)
    qualifying_orders      = order_size[order_size >= 2].index
    order_item             = order_item[order_item.index.isin(qualifying_orders)]

    print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))
    print("Remaining order_item: {:21d}".format(len(order_item)))


    # 対象外のitemとレシートを除外した後、もう一回支持率を計算する
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # 対象となる商品で、ペアを作る
    item_pair_gen          = get_item_pairs(order_item)

```

## 補足

### 参考資料
相関ルールのロジックと解説 https://www.kaggle.com/datatheque/association-rules-mining-market-basket-analysis  
yieldの使い方  
https://www.sejuku.net/blog/23716  
isinの使い方  
http://sinhrks.hatenablog.com/entry/2014/11/15/230705  
