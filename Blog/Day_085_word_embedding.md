# 85日目 word embedding

本日の目標は
1. word embeddingの目的
2. Word embeddingのアルゴリズム

## Step 1: Word embeddingの目的
目的：機械学習のアルゴリズムが理解できる特徴(feature)に変化する。  
Word Embedding では，同じ単語の要素すべてが同じ空間内に配置されるため，要素の特徴量は常に同じ長さのベクトルで表現される。  
このため，機械学習アルゴリズムに容易に投入することができる。  

## Step 2:　Word embeddingのアルゴリズム

### One-hot 表現
全ての単語を一つのベクターに入れ、存在1、不存在0の形で表現する。  
単語が存在するかしないかのみ検知できる。  

### 共起関係の利用(分散仮説)
count-based な手法：同じ文脈な中、単語の出現頻度を数える
predictive な手法：前後の文脈の単語のベクトルから目的の単語ベクトルを予測する

## 参考資料
http://www.orsj.or.jp/archive2/or62-11/or62_11_717.pdf  
