# 31日目 Principal Component Analysis Day31 Principal Component Analysis

本日の目標は
1. PCAのロジック理解
2. 簡単な次元圧縮を実装

## Step 1: PCAのロジック理解
全てのデータをテーブルで展開したら、各列が一つの次元になる。
全ての次元を分析の対象にすると、時間と資源の消耗が多くなるし、解説が難しくなる。
次元と次元の間に、相関性が存在する可能性がある。
N個の次元をK(K<N)個の次元に投影することにより、相関性を削減し、分析の効率が高まる。
教師なし分類問題でも使える。

## Step 2: 簡単な次元圧縮を実装
```python
from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np

# ランダムでデータ10個を生成する
dataset = datasets.make_blobs(n_samples=10)

# featureのみ残し、ラベルを外す（2軸のデータを生成された）
X = dataset[0]

# 生成したデータをラベルつける
labels = range(1, 11)  
plt.figure(figsize=(10, 7))  
plt.subplots_adjust(bottom=0.1)  
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):  
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()  

from scipy.cluster.hierarchy import dendrogram, linkage  

# linkage functionを使って、サンプルポイント間の距離を計算した後、dendrogramで出力できる形に変換
linked = linkage(X, 'single')

labelList = range(1, 11)

plt.figure(figsize=(10, 7))  

# dendrogram出力のfunction
dendrogram(linked,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()  

from sklearn.cluster import AgglomerativeClustering

# AgglomerativeClusteringでモデルを作成
# n_clustersはクラスターの数；affinityは最小距離の計算方法；linkageはクラスター間の関連性の計算方法
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(X)  

plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')  
plt.show()  


```
## Step 3: Agglomerative Hierarchical Clusteringの実装
```python

from sklearn.cluster import AgglomerativeClustering

# AgglomerativeClusteringでモデルを作成
# n_clustersはクラスターの数；affinityは最小距離の計算方法；linkageはクラスター間の関連性の計算方法
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(X)  

plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')  
plt.show()  

```

## 補足

### 参考資料
kmeansの実装 https://blog.amedama.jp/entry/2017/03/19/160121  
https://www.dogrow.net/python/blog33/  
https://qiita.com/Morio/items/0fe3abb58fcaff229f3d  
