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

def train(X, y, W, B, alpha, max_iters):
    '''
    Performs GD on all training examples,
    X: Training data set,
    y: Labels for training data,
    W: Weights vector,
    B: Bias variable,
    alpha: The learning rate,
    max_iters: Maximum GD iterations.
    '''
    dW = 0 # Weights gradient accumulator
    dB = 0 # Bias gradient accumulator
    m = X.shape[0] # No. of training examples
    for i in range(max_iters):
        dW = 0 # Reseting the accumulators
        dB = 0
        for j in range(m):
            # 1. Iterate over all examples,
            # 2. Compute gradients of the weights and biases in w_grad and b_grad,
            # 3. Update dW by adding w_grad and dB by adding b_grad,
         W = W - alpha * (dW / m) # Update the weights
         B = B - alpha * (dB / m) # Update the bias

    return W, B # Return the updated weights and bias.
```

## 補足

### 参考資料
Gradient Decent　https://hackernoon.com/gradient-descent-aynk-7cbe95a778da  
Gradient Decent explanation in Trad-Chinese https://medium.com/@arlen.mg.lu/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E8%AC%9B%E4%B8%AD%E6%96%87-gradient-descent-b2a658815c72  
Gradient Decent explanation in Trad-Chinese
https://ithelp.ithome.com.tw/articles/10193521
