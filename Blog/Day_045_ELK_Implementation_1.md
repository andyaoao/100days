# 45日目 ELK Implementation Day45 ELK_Implementation

本日の目標は
1. Logstachのdataをelasticsearchに送る
2. elasticsearchでデータを確認
3. kibanaでvisulization

## Step 1: Logstachの起動
Logstachのconfig fileをメンテし、

```java
// Sample Logstash configuration for creating a simple
// Beats -> Logstash -> Elasticsearch pipeline.

// fileAPIを使ってpathのcsvを指定します。
// 起動したタイミングで読み込みたいのでbegining (デフォルトはend)
// どこまで投入できたかどうかを覚えておくログを吐き出すパス(今回は設定しない)
input {
  file {
    path => "~/Users/andyaoao/Documents/Project/100days/Datasets/50_Startups.csv"
    start_position => "beginning"
    sincedb_path => "/dev/null"
  }
}

// 「,」でくぎられたところでカラムには一列目のセルの名前を入れます
// mutateでカラムの型のタイプを指定したり、加工したり、フィールドに入るデータを制御したりすることができます。

filter {
  csv {
    separator => ","
    columns => ["R&D","Spend","Administration","Marketing","Spend","State","Profit"]
  }
}

// ローカルの9200ポートでelasticsearchを起動しているのでhostsを向けます
// indexにはRDBMSでいうDB名を指定し
// document_typeにはRDBMSでいうテーブル名を指定します
output {
  elasticsearch {
    hosts => "http://localhost:9200"
    index => "wp_products"
    document_type => "products"
  }
  stdout {}
}
```

## Step 2: elasticsearchでデータを確認

curl -XGET指令でデータを確認する

```console
curl -XGET "http://localhost:9200/wp_products/products/_search?pretty"

```

## Step 3: kibanaでvisulization

http://127.0.0.1:5601/
