import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

#day 1 start

dataset = pd.read_csv('./Datasets/Data.csv')

# 最後の列はデータセットYとし、それ以外は、データせっとXとする
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , -1].values

# データセットXに対し、欠損値を補完する。
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

#day 2 start

# データセットXの１列名をカテゴリデータをコード化
labelencoder_X = LabelEncoder()
labelencoder_X.fit(X[ : , 0 ])
X[ : , 0] = labelencoder_X.transform(X[ : , 0])

# カテゴリ変数をダミー変数化
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Yをコード化し、本列を置換する。
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)

# Training set と　Testing set を分割
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

# Xのtraining set と　test set をfeature Scaling 処理
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

print (X_train)
