# 43日目 Installing ELK 1 Day43 Installing ELK 1

本日の目標は
1. Logstachのインストール
2. elasticsearchのインストール


## Step 1: Logstachのインストール
java 8 が必要。  
logstashインストール時に下記のエラーが発生した。  
Unrecognized VM option 'UseParNewGC'  

## Step 2: elasticsearchのインストール
cd elasticsearch-X.X.X/bin  
./elasticsearch  
下記出たら成功
[時間][INFO ][node                     ] [Paibo] started


## 補足: 古いJAVAバージョンの消し方
バージョンを確認
/usr/libexec/java_home -V
指定するjdkを削除
sudo rm -rf /Library/Java/JavaVirtualMachines/jdk1.8.0_92.jdk

### 参考資料
logstash install  
https://stackoverflow.com/questions/49623648/logstash-with-java10-get-error-unrecognized-vm-option-useparnewgc  
古いjavaの消し方  
https://qiita.com/okoshi/items/8ef75fb0104f55fd1a3c  
