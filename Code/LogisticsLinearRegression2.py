import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

#iris-dataを読み込む
df = pd.read_csv('./Datasets/iris-data.csv')

# 欠損値検知と修正
# column petal_width_cmに欠損値5個存在することを検知できた
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 150 entries, 0 to 149
# Data columns (total 5 columns):
# sepal_length_cm    150 non-null float64
# sepal_width_cm     150 non-null float64
# petal_length_cm    150 non-null float64
# petal_width_cm     145 non-null float64
# class              150 non-null object
# dtypes: float64(4), object(1)
# memory usage: 5.9+ KB

# column petal_width_cmに欠損値が存在するレコードを削除
df = df.dropna(subset=['petal_width_cm'])

# 変数間の関係性を図で出力する
sns.pairplot(df, hue='class', size=2.5)
# plt.show()

# classの名称が間違っていることを発見。修正する
df['class'].replace(["Iris-setossa","versicolor"], ["Iris-setosa","Iris-versicolor"], inplace=True)

# logistics regressionのoutputは2択なので、一旦2択のみ残す
final_df = df[df['class'] != 'Iris-virginica']

# outlier存在するかどうかを図でチェック
sns.pairplot(final_df, hue='class', size=2.5)
# Iris-setosaのsepal_width_cmにoutlierが存在する
# Iris-versicolorのsepal_height_cmにoutlierが存在する

# 簡単処理のため、outlierが存在するデータを一旦除外する・
final_df = final_df.drop(final_df[(final_df['class'] == "Iris-setosa") & (final_df['sepal_width_cm'] < 2.5)].index)
final_df = final_df.drop(final_df[(final_df['class'] == "Iris-versicolor") & (final_df['sepal_length_cm'] < 1)].index)

sns.pairplot(final_df, hue='class', size=2.5)
# plt.show()

# outputに対し、コード化する
final_df['class'].replace(["Iris-setosa","Iris-versicolor"], [1,0], inplace=True)

# inputとoutputのdfを作成
inp_df = final_df.drop(final_df.columns[[4]], axis=1)
out_df = final_df.drop(final_df.columns[[0,1,2,3]], axis=1)

#　feature scaling
scaler = StandardScaler()
inp_df = scaler.fit_transform(inp_df)

#　data set分割
X_train, X_test, y_train, y_test = train_test_split(inp_df, out_df, test_size=0.2, random_state=42)

# Yをarray化
y_tr_arr = y_train.as_matrix()
y_ts_arr = y_test.as_matrix()

# LogisticRegressionモデルを作成
clf = LogisticRegression()
clf.fit(X_train, y_tr_arr)

# 予測結果
pred  =  clf.predict(X_test)
print (pred)
