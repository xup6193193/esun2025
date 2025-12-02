# README: Model（Random Forest Ensemble）

此程式接續 Preprocess 輸出，進行特徵工程與隨機森林 Ensemble 訓練，用於從 predict 名單中挑選最可疑帳戶。

## 輸入檔案
- acct_alert_view_export.csv  
- acct_predict_view_export.csv  
- acct_predict.csv  

## 功能流程
1. 特徵工程（可依 FLAG 分群）  
2. 隨機森林 Bagging 訓練多模型  
3. 融合機率並挑出前 N_TO_PREDICT_POSITIVE  
4. 匯出 submission_platform_*.csv、probabilities_*、feature_importance_*  

## 執行
```
python Model.py
```
