# 42日目 Understanding ELK 2 Day42 Understanding ELK 2

本日の目標は
1. 全文検索とは
2. Elasticsearchのデータ構造


## Step 1: 全文検索とは
複数のファイルの中、特定の文字列を検索する機能。  
document：ファイル(文書)  
単語：documentより抽出した文言  
id：単語にどのdocumentに存在しているflag  
単語の区切り方
N-Gram：N文字ずつ区切る　→　効率が良くない  
形態素解析：辞書をベースで、単語を区切る  

## Step 2: Elasticsearchのデータ構造
cluster -> node -> index -> type -> field  
index:database; type:table の関係性  

```java
// customerというデータベースを登録
curl -XPUT 'localhost:9200/customer'
// mapping用のjsonファイルを登録
curl -XPOST localhost:9200/customer -d @mapping.json
// typeを生成し、レコード追加
curl -XPUT 'localhost:9200/customer/external/1?pretty' -d '
{
  "name": "John Doe"
}'
// データベース削除
curl -XDELETE 'localhost:9200/customer?pretty'
// レコードのupdate
curl -XPOST 'localhost:9200/customer/external/1/_update?pretty' -d '
{
  "doc": { "name": "Jane Doe", "age": 20 }
}'

```


### 参考資料
全文検索  
https://speakerdeck.com/johtani/elasticsearchfalseshi-mefang?slide=53  
Elasticsearchのデータ構造  
http://code46.hatenablog.com/entry/2014/01/21/115620  
https://qiita.com/kws9/items/7695262be0befb94897f  
https://blog.shibayu36.org/entry/2016/09/05/110000  
