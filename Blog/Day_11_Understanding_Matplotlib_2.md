# 10日目Matplotlib検証　Day 10 Understanding Matplotlib

本日の目標は
1. subset設定（異なるsubsetに対して背景色を設定）
2. グラフの構成
3. KNNのtraining data を図で描く

## Step 1: subset設定
```python
import matplotlib.pyplot as plt

# Figure(一つの図)
fig = plt.figure()  
# Subplot(一つの図の中のsub図)　
# Subplot(211) 行x列x象限(1は左上)　
ax = fig.add_subplot(211)  # 2x2の図の中に、左上のsub図
ax1 = fig.add_subplot(212)  # 2x2の図の中に、右上のsub図
```

## Step 2: 背景色を設定
```python
# 図全体の背景色と背景透明度を設定
fig.patch.set_facecolor('blue')  
fig.patch.set_alpha(0.5)  
# 2x2の図の中に、左上のsub図に対しての背景色と背景透明度を設定
ax.patch.set_facecolor('green')
ax.patch.set_alpha(0.3)
# 2x2の図の中に、右上のsub図に対しての背景色と背景透明度を設定
ax1.patch.set_facecolor('green')  
ax1.patch.set_alpha(0.3)  
```

## Step 3: KNNのtraining data を図で描く
```python
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
```

## 補足

### 参考資料
matplotlibの背景色設定　http://kaisk.hatenadiary.com/entry/2014/11/30/163016  
KNN plot 説明 https://ameblo.jp/cognitive-solution/entry-12289785974.html  
