# 17日目Neural Network 2　Day 16 Neural Network 2

本日の目標は
1. データを取り込みから整理まで
2. output layerの設定
3. トレーニングのプロセスを理解

## Step 1: データを取り込みから整理まで

```python
# numpyが用意しているデータをロード
digits = load_digits()
plt.gray()
plt.matshow(digits.images[2])
# plt.show()

# 各数値は8X8のpixel(データ)に構成されている
# 64個0-15の数値がある
digits.data[0,:]

# Activationには、0-1の数値を使用するので、feature scaling
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)

# データを分割
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

```

## Step 2: output layerの設定

```python

# output layerの設定
def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect
y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)
# print (y_train[1], y_v_train[1])

```

## Step 3: トレーニングのプロセスを理解

トレーニングのプロセスは下記：
Randomly initialise the weights for each layer W(l)
While iterations < iteration limit:
1. Set ΔW and Δb to zero
2. For samples 1 to m:  
a. Perform a feed foward pass through all the nl layers. Store the activation function outputs h(l)  
b. Calculate the δ(nl) value for the output layer  
c. Use backpropagation to calculate the δ(l) values for layers 2 to nl−1  
d. Update the ΔW(l) and Δb(l) for each layer  
3. Perform a gradient descent step using:

W(l)=W(l)–α[1mΔW(l)]  
b(l)=b(l)–α[1mΔb(l)]


## 補足

### 参考資料
Neural Networks Tutorial – A Pathway to Deep Learning   http://adventuresinmachinelearning.com/neural-networks-tutorial/#structure-ann  
Gradient descent, how neural networks learn | Deep learning, chapter 2  	https://www.youtube.com/watch?v=IHZwWFHWa-w  
