# 28日目Kmeans Clustering 1 Day28 Kmeans Clustering 1

本日の目標は
1. unsupervised learningの目的を理解する
2. sklearnのkmeansを実装してみる

## Step 1: TensorBoardでベストプラクティスを探す
ラベルを持っていない時使用する。  
目的は、データの中から、情報を整理する。類似のデータポイントを集めた上で、論理性をづける。  

## Step 2: sklearnのkmeansを実装してみる

```python
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# ランダムでデータを生成する
dataset = datasets.make_blobs()

# featureのみ残し、ラベルを外す（2軸のデータを生成された）
features = dataset[0]
# targets = dataset[1]

# クラスタ数（何分類まで分類されると定義する）
N_CLUSTERS = 5

# クラスタリングモデルを生成する
cls = KMeans(n_clusters=N_CLUSTERS)

# 予測する
pred = cls.fit_predict(features)

# 各分類をラベルごとに色付けして表示する
for i in range(N_CLUSTERS):
    labels = features[pred == i]
    plt.scatter(labels[:, 0], labels[:, 1])

# クラスタのcentralを他の色で描く
centers = cls.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=100,
            facecolors='none', edgecolors='black')

plt.show()
```

## 補足

### 参考資料
kmeansの実装 https://blog.amedama.jp/entry/2017/03/19/160121  
