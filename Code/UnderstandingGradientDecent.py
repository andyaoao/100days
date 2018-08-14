import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#day 7 start

#5列のデータセットを読み込む
dataset = pd.read_csv('./Datasets/Social_Network_Ads.csv')

# datasetの欠損値チェック
dataset.isnull().any()
# User ID            False
# Gender             False
# Age                False
# EstimatedSalary    False
# Purchased          False
# dtype: bool

sns.pairplot(dataset, hue='Gender', size=2.5)


# データ分割
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#
# # feature scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
#
# # 詳細のデータ前処理はday1にご参考
# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)
#
# # 作成されたモデルで予測
# y_pred = classifier.predict(X_test)
#
# print (y_pred)
