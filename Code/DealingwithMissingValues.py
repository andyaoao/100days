import pandas as pd

#day 6 start

dataset = pd.read_csv('./Datasets/Data.csv')

# 行か列全体に対して、欠損しているかどうか
print(dataset.isnull().all())
# 行か列の中に、欠損値が存在するかないか
print(dataset.isnull().any())
# 列ごとの欠損値個数合計
print(dataset.isnull().sum())

# 行全体が欠損値の場合、除外
print(dataset.dropna(how='all'))
# 列全体が欠損値の場合、除外
print(dataset.dropna(how='all', axis=1))

# 欠損値が一つでも含まれる行が削除される(二つ同意)
print(dataset.dropna(how='any'))
print(dataset.dropna())

# 個数(下記は3)を指定すると含まれる欠損値の数に応じて行を削除
print(dataset.dropna(thresh=3))

# pandas　で欠損値を置換
# 平均値で置き換える
print(dataset.fillna(dataset.mean()))
# 前の値で置き換える
print(dataset.fillna(method='ffill'))
# 後の値で置き換える
print(dataset.fillna(method='bfill'))
