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


# # orderの数(レシート数)を計算
# def get_item_pairs(order_item):
#     order_item = order_item.reset_index().as_matrix()
#     for order_id, order_object in groupby(order_item, lambda x: x[0]):
#         item_list = [item[1] for item in order_object]
#
#         for item_pair in combinations(item_list, 2):
#             yield item_pair
#

# # Returns frequency and support associated with item
# def merge_item_stats(item_pairs, item_stats):
#     return (item_pairs
#                 .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
#                 .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))
#
#
# # Returns name associated with item
# def merge_item_name(rules, item_name):
#     columns = ['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB',
#                'confidenceAtoB','confidenceBtoA','lift']
#     rules = (rules
#                 .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
#                 .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))
#     return rules[columns]


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

    # # 各ペアの
    # item_pairs              = freq(item_pair_gen).to_frame("freqAB")
    # item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100
    #
    # print("Item pairs: {:31d}".format(len(item_pairs)))
    #
    #
    # # Filter from item_pairs those below min support
    # item_pairs              = item_pairs[item_pairs['supportAB'] >= min_support]
    #
    # print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))
    #
    #
    # # Create table of association rules and compute relevant metrics
    # item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    # item_pairs = merge_item_stats(item_pairs, item_stats)
    #
    # item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
    # item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
    # item_pairs['lift']           = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])
    #
    #
    # # Return association rules sorted by lift in descending order
    # return item_pairs.sort_values('lift', ascending=False)


rules = association_rules(orders, 0.01)
# item_name   = pd.read_csv('../input/products.csv')
# item_name   = item_name.rename(columns={'product_id':'item_id', 'product_name':'item_name'})
# rules_final = merge_item_name(rules, item_name).sort_values('lift', ascending=False)
# display(rules_final)
