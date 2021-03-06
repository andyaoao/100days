import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
# print(X_train, X_test, y_pred)

#day 11 start

def plot_decision_regions(X, y, classifier, resolution=0.01):

    # ポイントのマーク、色、背景色を定義する
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # figureのボーダーを定義する(等差数列を作って、等高線のgridを作るため)
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))

    # 等差数列に変換した値をもう一回分類機を通らせる
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # 等高線を描く
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # training set のポイントを散布図として描く
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),marker=markers[idx], label=cl)

plot_decision_regions(X_train, y_train, classifier=classifier)

plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
