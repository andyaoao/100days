import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#day 9 start

#5列のデータセットを読み込む
dataset = pd.read_csv('./Datasets/Social_Network_Ads.csv')
# 3, 4列をXとして格納
X = dataset.iloc[:, [2, 3]].values
# 最後の列はY(1行)として格納
y = dataset.iloc[:, 4].values

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# minkowskiはsum(|x - y|^p)^(1/p)
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# 予測結果
y_pred = classifier.predict(X_test)
