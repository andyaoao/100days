# 16日目Neural Network 1　Day 16 Neural Network 1

本日の目標は
1. Neural Networkの基礎を説明できる

## Step 1:
But what is a Neural Network? | Deep learning, chapter 1  https://www.youtube.com/watch?v=aircAruvnKk&t=7s  
Neural NetwÂork
Neurons : A thing with a number which is between 0 to 1 -> a function which has a output between 0 and 1.  
Activation : The number in a neuron.  
Input Layer : Raw data was split into multiple neurons.  
Hidden Layer : multiple neurons is existing in one Hidden layer.  
Output Layer : Decide the output (pridiction).  

Logic :
Output Layerから逆算で、最後のoutputはどういったelementに決められるか、とどんどん深掘りして、最後は、Input Layerのinput elementになる。  

Neuron Activationの計算：
For each neuron : sigmoid(w1a1 + w2a2 + w3a3 .... + bias)  
w = weight  
a = neuron activation from previous Layer  
bias = it is that output of the neural net when it has absolutely zero input.
** each neuron activation is a value between 0 to 1, so the function is sigmoid function  

何も勉強させる？
各neuron(function)のweightとbias。
weightとbiasの調整で、モデルの予測精度をあげる。

## 補足

### 参考資料
What is bias in artificial neural network? https://www.quora.com/What-is-bias-in-artificial-neural-network  
Decision Trees: “Gini” vs. “Entropy” criteria	https://www.garysieling.com/blog/sklearn-gini-vs-entropy-criteria8-47c94095171  
