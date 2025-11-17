# README

## 📌 專案簡介

此專案包含兩大模組：

1.  **資料前處理 (`Preprocess.py`)**
    -   從原始交易資料 (*acct_transaction.csv*)
        生成模型可使用的訓練/預測視圖：
        -   `acct_alert_view_export.csv`
        -   `acct_predict_view_export.csv`
    -   依據 SQL-like 規則，產生四種交易關聯區段： `FROM`, `TO`,
        `FROM_NEXT`, `TO_BEFORE`
2.  **模型訓練與預測 (`Model.py`)**
    -   對帳戶做特徵工程（Pivot/Combine）
    -   使用多模型 Bagging + Undersampling（N=240）
    -   產生預測標籤：只標記機率最高的前 N=240 筆為 1
    -   自動輸出平台提交檔、機率檔、特徵重要性檔案

## 📁 檔案結構

    Preprocess.py
    Model.py
    (官方資料）
    acct_transaction.csv
    acct_alert.csv
    acct_predict.csv

## 🧩 模組 1：Preprocess（生成交易視圖）

### 功能摘要

`Preprocess.py` 用來模擬 SQL 中的多層
join/union，將帳戶與交易資料結合，依四種邏輯產生標記(`FLAG`)：

  FLAG        說明
  ----------- ----------------------
  FROM        from_acct = 帳戶
  TO          to_acct = 帳戶
  FROM_NEXT   下一層關聯 from → to
  TO_BEFORE   反向關聯 to → from

### 輸出欄位

    FLAG, ACCT, FROM_ACCT_TYPE, TO_ACCT_TYPE, IS_SELF_TXN,
    TXN_AMT, TXN_DATE, TXN_TIME, CURRENCY_TYPE, CHANNEL_TYPE

------------------------------------------------------------------------

## 🧩 模組 2：Model（特徵工程＋Ensemble 預測）

### 功能摘要

此模組執行整個特徵工程與模型預測 pipeline：

-   FLAG split（自動偵測所有 FLAG）
-   全面統計特徵（交易金額/通路/幣別/HOUR...）
-   特徵群組化：可啟用/停用
-   Bagging Ensemble（240 RF 模型）
-   Undersampling（1:1）
-   只輸出機率最高 240 筆為 1
-   自動輸出提交檔與特徵重要性

------------------------------------------------------------------------

## ⚙️ 主要參數

    FLAG_OPEN = 'no'
    OUTPUT_DIR = 'sklearn_rf_next_before'
    N_TO_PREDICT_POSITIVE = 240
    N_ESTIMATORS_ENSEMBLE = 240

------------------------------------------------------------------------

## ▶️ 執行方式

### Step 1：生成視圖

    python Preprocess.py

### Step 2：執行模型

    python Model.py

------------------------------------------------------------------------

## 📤 輸出結果

-   submission_platform\_\*.csv
-   submission_probabilities\_\*.csv
-   feature_importance\_\*.csv
-   parameters\_\*.csv
