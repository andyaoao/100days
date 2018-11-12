# 85日目 word embedding 2 Word2vec

本日の目標は
1. Word2vecの概念

## Step 1: Word2vecの概念
f(x) = y の関数で、xとy両方ともword、xとyは同じ文に存在するとき、人間が理解できる文言になれるかどうかの関数。  
全ての単語がvectorを持っている、任意の二つの単語(vector)は距離を計算できる。  
a, b, c, d 4つの単語があった場合、aとbの距離はcとdが同じの場合、a, b, cが知って入れば、dがわかる。  

### Skip-gram
単語　->　近くの文言を予測する
inputは単語１個（one hot方法のであれば、0,1のvectorになる）、1個のhidden layer、outputは複数。

### CBOW
近くの文言　->　単語を予測する
inputは単語複数個（平均などの方法でまとめる）、1個のhidden layer、outputは１個。

## Step 2:　WWord2vecのoutput

### similar
一つの単語をinputしたら、各単語に数値が出る。  
その数値は近い単語の距離を表している。（高ければ、近い）  

### calculation
例１：king - man + woman = queen
例２：Paris - France + Italy = Roma

kingとmanのvector様子は、womanの対応単語を予測できる。

## 参考資料
https://zhuanlan.zhihu.com/p/26306795  
https://qiita.com/teresa/items/b5dc3b0a25aea49f78a1  
