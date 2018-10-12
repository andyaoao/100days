# 65日目 LTSM 1 Day65 LTSM 1

本日の目標は
1. 時間ー文字列の処理
2. 分析案5を考案

## Step 1: 時間ー文字列の処理
strptime：文字列を時間にparse  
```python
from datetime import datetime
d = datetime.strptime("1991-11-05", "%Y-%m-%d")
```

strftime：時間を文字としてparse
```python
from datetime import datetime
d = datetime(1991, 11, 5)
print(d.strftime("%Y-%m-%d"))  
```

## Step 2: 分析案5を考案
priceは考えていないから、LTSMにprice要素を追加


##　参考資料
https://docs.pyq.jp/help/quest/quest_help_strftime_strptime.html  
