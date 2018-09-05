# 28日目Kmeans Clustering 2 Day28 Kmeans Clustering 2

本日の目標は
1. KMeans clusteringを自分で実装してみる

## Step 1: KMeans clusteringを自分で実装してみる

```python
# KMeans clusteringを自分で実装してみる
class KMeans(object):

    # クラスを定義する
    # クラスター数とiterationの回数を定義する
    def __init__(self, n_clusters=2, max_iter=300):

        self.n_clusters = n_clusters
        self.max_iter = max_iter

        self.cluster_centers_ = None

    def fit_predict(self, features):

        # featureの中より、ランダムで中心を選択する(個数は、initで定義されたクラスタ数)
        feature_indexes = np.arange(len(features))
        np.random.shuffle(feature_indexes)
        initial_centroid_indexes = feature_indexes[:self.n_clusters]
        self.cluster_centers_ = features[initial_centroid_indexes]

        # 予測用のarrayは一回zeroで初期化する
        pred = np.zeros(features.shape)

        # クラスタリングをアップデートする
        for _ in range(self.max_iter):
            # pはfeatureの中の各point、centroidは中心点
            # 一つのポイントは各クラスターの中心とそれぞれ計算する。
            # 計算結果を比較して、最小値の方は、このポイントの所属クラスター。
            new_pred = np.array([
                np.array([
                    self._euclidean_distance(p, centroid)
                    for centroid in self.cluster_centers_
                ]).argmin()
                for p in features
            ])

            if np.all(new_pred == pred):
                # 更新前と内容が同じなら終了
                break

            pred = new_pred

            # 各クラスタごとに中心を再計算する
            self.cluster_centers_ = np.array([features[pred == i].mean(axis=0)
                                              for i in range(self.n_clusters)])

        return pred

    # 距離を計算用のfunction
    def _euclidean_distance(self, p0, p1):
        return np.sum((p0 - p1) ** 2)
```

## 補足

### 参考資料
kmeansの実装 https://blog.amedama.jp/entry/2017/03/19/160121  
https://www.dogrow.net/python/blog33/  
https://qiita.com/Morio/items/0fe3abb58fcaff229f3d  
