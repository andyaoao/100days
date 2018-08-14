# 7日目Gradient Decent の理解　Day 7 Understanding Gradient Decent

本日の目標は
1. Gradient Decent の理解
2. Gradient Decent の実現ステップ
3. Codingで実現

## Step 1: Gradient Decent の目的

1. Gradient Decentは、予測モデル生成の際に、エラーを最小化するアルゴリズムである。
2. Errorを最小化するために、ErrorをモニタリングするFunctionを作る。Cost Functionと呼ばれる。
3. Cost Functionを通して、Costを最小化 = Errorの最小化。
4. Cost Functionを図で描いてみたら、縦軸はerror、横軸はweight（仮に一つの説明変数のみ）、関数の最低点を探すは目的である。
5. 普通のMLモデルは複数のweight(説明変数)が存在しているはずである。その際に、図で表現することが難しい。


## Step 2: Gradient Decent の実現ステップ
1. Cost Functionのエラー判定はいつくか種類があるが、一番よく使われているのはMSEである。
2. Cost Functionでgradientを計算する。全てのweightのgradientとバイアスのgradientを計算し合計する。ΣdWi
3. 計算されたgradientをgradient accumulatorに合算する：dW
4. updated accumulators を計算する：合算したgradient accumulatorをtraining sampleに割る：(dW / m)
6. 新しいweightとバイアスを更新する：W = W - alpha * (dW / m)

## Step 3: Codingで実現
```python

# コストの計算を別出し
def compute_cost(features, values, weight):
    """
    Compute the cost of a list of parameters, weight, given a list of features
    (input data points) and values (output data points).
    """
    # テストのデータ数
    m = len(values)
    # MSEの合計を計算
    sum_of_square_errors = np.square(np.dot(features, weight) - values).sum()
    # コスト（MSEの合計 / テストデータ数の2倍
    cost = sum_of_square_errors / (2*m)

    return cost

# gradient descentの計算
def gradient_descent(features, values, weight, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    """

    # コストの推移を記録する
    cost_history = []

    # training iterationsは実行の回数
    for i in range(0, num_iterations):
        # 毎回計算したコストを陣列に保存
        cost_history.append(compute_cost(features, values, weight))
        # 関数での計算結果
        hypothesis = np.dot(features, weight)
        # 関数の計算結果とテスト数値の差はロスと扱う
        loss = hypothesis - values
        # gradientを計算
        gradient = np.dot(features.transpose(), loss) / len(values)
        # gradientにより、weightを修正
        weight = weight - alpha*gradient
    return weight, cost_history
```

## 補足

### 参考資料
Gradient Decent　https://hackernoon.com/gradient-descent-aynk-7cbe95a778da  
Gradient Decent explanation in Trad-Chinese https://medium.com/@arlen.mg.lu/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E8%AC%9B%E4%B8%AD%E6%96%87-gradient-descent-b2a658815c72  
Gradient Decent explanation in Trad-Chinese
https://ithelp.ithome.com.tw/articles/10193521
