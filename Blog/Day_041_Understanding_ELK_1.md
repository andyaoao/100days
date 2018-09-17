# 41日目 Understanding ELK 1 Day41 Understanding ELK 1

本日の目標は
1. ELKの簡単理解
2. Logstash
3. Elasticsearch
4. Kibana


## Step 1: ELKの理解
ELKは三つのopen source packageの略称、データの収集、整理、加工、分析など一連のタスクができる
Logstash：データを扱う　　
Elasticsearch：データを加工し投入する　　
Kibana：投入したデータを可視化する　　

## Step 2: Logstash
Logstashの用途は、データを扱うこと。  
configuration fileが必要。その中に、input、filter、outputが定義される(下記例)
簡単に加工したoutputはelasticsearchにIFしやすい  

```java
input { stdin {} }

filter {
  grok {
    match => {
      "message" => '%{HTTPDATE:date} %{IP:ip} '
    }
  }
}

output {
  stdout {
    codec => rubydebug
  }
}

```
## Step 3: Elasticsearch
NoSQLのデータベース。  
全てのCRUD操作はjsonで実現する。  
複数のnodeで一つのclusterを構成する。  
全てのnodeにshardが保存されている。  
shardにはoriginalとreplicaがある。分散で、各nodeに保存してある。  

## Step 4: Kibana
BIツール、visulizarion、dashboardを用意してある。  


### 参考資料
ELKの紹介  
https://engineers.weddingpark.co.jp/?p=1876  
logstashのapi  
https://www.elastic.co/guide/en/logstash/current/filter-plugins.html 　
