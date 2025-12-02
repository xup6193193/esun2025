# README: Preprocess（acct_xxx_view_export 產生器）

此程式負責將銀行交易資料（acct_transaction.csv）與 alert/predict 名單進行關聯分析，並依照 SQL 商業邏輯模擬輸出兩個 view：

- `acct_predict_view_export.csv`
- `acct_alert_view_export.csv`

產生的資料會提供後續 Model（隨機森林）進行特徵工程與訓練。

## 輸入檔案
- acct_transaction.csv  
- acct_alert.csv  
- acct_predict.csv  

## 功能流程摘要
1. 依帳號排序建立 group_p  
2. 模擬 SQL UNION（FROM / TO / FROM_NEXT / TO_BEFORE）  
3. 依 group_p join 回來源資料  
4. 欄位大寫化  
5. 匯出 acct_predict_view_export.csv 與 acct_alert_view_export.csv  

## 執行
```
python Preprocess.py
```
