# 35日目 ETLの理解 Day35 Understanding ETL

本日の目標は
1. ETLの理解
2. Airflowの環境構築

## Step 1: ETLの理解
抽出：データ来ることを検知した時、ソースよりデータを抽出する（トリガーは別途定義される）  
Extract: this is the step where sensors wait for upstream data sources to land (e.g. a upstream source could be machine or user-generated logs, relational database copy, external dataset … etc). Upon available, we transport the data from their source locations to further transformations.  

変換：抽出してきたデータをフィルターリング、累計などの処理  
Transform: This is the heart of any ETL job, where we apply business logic and perform actions such as filtering, grouping, and aggregation to translate raw data into analysis-ready datasets. This step requires a great deal of business understanding and domain knowledge.

ロード：整理できたデータをターゲット格納先に保存する  
Load: Finally, we load the processed data and transport them to a final destination. Often, this dataset can be either consumed directly by end-users or it can be treated as yet another upstream dependency to another ETL job, forming the so called data lineage.

## Step 2: Airflowの環境構築

conda create -n airflow-tutorials python=3.6 -y  
source activate airflow-tutorials  
pip install "apache-airflow[crypto, slack]"  
ここで失敗になっている。調査中  
