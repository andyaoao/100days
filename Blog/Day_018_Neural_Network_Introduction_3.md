# 18日目Neural Network 3　Day 18 Neural Network 3

本日の目標は
1. トレーニングのプロセスの実装

## Step 1: トレーニングのプロセスの実装

```python
# create neural Network
# 3 layer with 64 30 10 neurons
nn_structure = [64, 30, 10]

# sigmoid functionの用意
def f(x):
    return 1 / (1 + np.exp(-x))
def f_deriv(x):
    return f(x) * (1 - f(x))

# ランダムでweightとbiasを初期化
def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b

# weightとbiasの初期値はzeroを設定する
def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b

# perform a feed forward pass through the network
def feed_forward(x, W, b):
    h = {1: x}
    z = {}
    for l in range(1, len(W) + 1):
        # lは 1 のであれば、input layer, node_in(input)はそのまま
        if l == 1:
            node_in = x
        # lは 1 以外のであれば、node_inは前のNodeのoutputだ
        else:
            node_in = h[l]
        # outputを計算する
        z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)
        h[l+1] = f(z[l+1]) # h^(l) = f(z^(l))
    return h, z

# gradient decentのcost functionのdeltaを計算するため
def calculate_out_layer_delta(y, h_out, z_out):
    # delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
    return -(y-h_out) * f_deriv(z_out)

def calculate_hidden_delta(delta_plus_1, w_l, z_l):
    # delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
    return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)

# neural networkのmain function
# alphaは学習率
def train_nn(nn_structure, X, y, iter_num=30, alpha=0.25):
    W, b = setup_and_init_weights(nn_structure)
    cnt = 0
    m = len(y)
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        if cnt%1000 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))
        tri_W, tri_b = init_tri_values(nn_structure)
        avg_cost = 0
        for i in range(len(y)):
            delta = {}
            # feed_forwardでgradient descent 計算お用のhとzをだす
            h, z = feed_forward(X[i, :], W, b)
            # backpropagation方法でcostを計算する
            for l in range(len(nn_structure), 0, -1):
                # output layoutのであれば、calculate_out_layer_deltaでcotを計算する
                if l == len(nn_structure):
                    delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
                    avg_cost += np.linalg.norm((y[i,:]-h[l]))
                else:
                # output layout以外のであれば、calculate_hidden_layer_deltaでcotを計算する
                    if l > 1:
                        delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
                    # triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
                    tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))
                    # trib^(l) = trib^(l) + delta^(l+1)
                    tri_b[l] += delta[l+1]
        # gradient descent 方法で各layerのweightとbiasを調整(学習率により)
        for l in range(len(nn_structure) - 1, 0, -1):
            W[l] += -alpha * (1.0/m * tri_W[l])
            b[l] += -alpha * (1.0/m * tri_b[l])
        # 平均costを計算
        avg_cost = 1.0/m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1
    return W, b, avg_cost_func

W, b, avg_cost_func = train_nn(nn_structure, X_train, y_v_train)

```

## 補足

### 参考資料
Neural Networks Tutorial – A Pathway to Deep Learning   http://adventuresinmachinelearning.com/neural-networks-tutorial/#structure-ann  
Gradient descent, how neural networks learn | Deep learning, chapter 2  	https://www.youtube.com/watch?v=IHZwWFHWa-w  
