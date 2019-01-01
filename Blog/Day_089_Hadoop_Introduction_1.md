# 89日目 Hadoop_Introduction 1

本日の目標は
1. Hadoop の 目標、簡単なロジックを理解する

## Step 1: Hadoop の 目標
Big Dataのハンドリングを簡易にする。全てのデータを一つのサーバー上で保存するのではなく、いくつかのサーバー(Distributed Store)に保存する。  
Bigデータの課題：データ量、データの形（Structured data, Unstructured data）
Solution：MapReduce手法でデータを複製し、保存する。

## Step 2: Hadoop の ロジック

### Map Reduceのロジック
書き込み；データが保存される際に、複製し、複数のところで保存する。保存する際に、ソートをかける。
読み込み：cluster（一連の端末）に対して検索をかける。一つの端末ではないため、平行で処理できる。

### mapper



### reducer







## 参考資料
https://jp.talend.com/resources/what-is-mapreduce/  
http://oss.infoscience.co.jp/hadoop/common/docs/r0.20.1/mapred_tutorial.html  
