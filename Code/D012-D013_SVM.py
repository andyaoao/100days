import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV

#day 12 start

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

# Fitting SVM to the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# 予測結果
y_pred = classifier.predict(X_test)

# cunfusion matrixを作る
cm = confusion_matrix(y_test, y_pred)

# Day 11で作った等高線function
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

# Training set を図で表現する
plot_decision_regions(X_train, y_train, classifier=classifier)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
# plt.show()

# Test set を図で表現する
plot_decision_regions(X_test, y_test, classifier=classifier)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
# plt.show()

# day 13 start

# 探索するパラメータを設定
param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
# 評価関数を指定
scores = ['accuracy', 'precision', 'recall']

# 各評価関数ごとにグリッドサーチを行う
for score in scores:
    # SVCをベースでgrid searchを行う
    clf = GridSearchCV(SVC(C=1), param_grid, cv=5, scoring=score, n_jobs=-1)  # n_jobs: 並列計算を行う（-1 とすれば使用PCで可能な最適数の並列処理を行う）
    clf.fit(X_train, y_train)

    # 最適なパラメータを表示
    print (clf.best_estimator_)


# grid search後、もう一度最適なパラメータで予測する
y_pred_1 = clf.predict(X_test)
