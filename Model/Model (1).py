import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
# 匯入 scikit-learn 相關套件
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import resample
import warnings
from functools import reduce # 為了合併多個 DataFrame

# ------------------------------------------------------------------------
# --- 1. 參數設定 (Parameters) ---
# --- (已根據您的最新需求更新) ---
# ------------------------------------------------------------------------

# 'yes': 欄位拆分為 FROM_, TO_, FROM_NEXT_ 等 (Pivoted)
# 'no':  欄位合併 (Combined)
FLAG_OPEN = 'no' 

# 輸出子目錄 (儲存 .csv 檔)
OUTPUT_DIR = 'sklearn_rf_next_before' # <-- 已更新為您指定的值

# **策略**：我們要預測多少筆資料為 "1"
N_TO_PREDICT_POSITIVE = 240 

# **V6 策略**：我們要訓練多少個模型
N_ESTIMATORS_ENSEMBLE = 240 

# ========================================================================
# --- 【需求 1 v2】參數化的特徵群組 ---
# 
# 決定要啟用 (True) 或禁用 (False) 哪些特徵群組。
#
FEATURE_GROUPS_TO_USE = {
    # 基礎統計: total_txn_count, txn_frequency
    'txn_stats': True,
    
    # 交易日期統計: txn_date_max, txn_date_min, txn_date_range
    'txn_date_stats': False,
    
    # 交易金額統計: txn_amt_mean, txn_amt_std, txn_amt_max, txn_amt_min
    'txn_amt_stats': True,
    
    # 是否自行交易 (IS_SELF_TXN) 的分佈
    'is_self_txn_dist': True,
    
    # TO_ACCT_TYPE 的分佈
    'to_acct_type_dist': True,
    
    # FROM_ACCT_TYPE 的分佈
    'from_acct_type_dist': True,
    
    # CHANNEL_TYPE (通路) 的分佈
    'channel_type_dist': True,
    
    # txn_hour (交易小時) 的分佈
    'txn_hour_dist': True,
    
    # currency_type (幣別) 的分佈
    'currency_type_dist': False
}
# ========================================================================


# ------------------------------------------------------------------------
# --- 2. 特徵工程函式 (Feature Engineering Function) ---
# (此區塊不需要修改，保持原樣)
# ------------------------------------------------------------------------

def _calculate_features(df, group_keys):
    """
    內部輔助函式：對傳入的 DataFrame 進行特徵計算。
    """
    
    if df.empty:
        print("  > 傳入的 DataFrame 為空，跳過計算。")
        return pd.DataFrame()
        
    print(f"  > 正在處理 {len(df)} 筆交易...")
    
    df = df.copy()
    
    df['TXN_AMT'] = pd.to_numeric(df['TXN_AMT'], errors='coerce')
    df['TXN_DATE'] = pd.to_numeric(df['TXN_DATE'], errors='coerce')
    
    df['TXN_TIME'] = df['TXN_TIME'].astype(str)
    df['txn_hour'] = pd.to_datetime(df['TXN_TIME'], format='%H:%M:%S', errors='coerce').dt.hour
    df['txn_hour'] = df['txn_hour'].fillna(-1).astype(int) 

    df['IS_SELF_TXN'] = df['IS_SELF_TXN'].astype(str)
    df['TO_ACCT_TYPE'] = df['TO_ACCT_TYPE'].astype(str)
    df['FROM_ACCT_TYPE'] = df['FROM_ACCT_TYPE'].astype(str)
    
    # 清理 currency_type，將 NaN 填充為 'UNK'
    df['currency_type'] = df['currency_type'].fillna('UNK').astype(str)
    
    df['CHANNEL_TYPE'] = df['CHANNEL_TYPE'].astype(str)
    df['CHANNEL_TYPE'] = df['CHANNEL_TYPE'].str.replace(r'^0(\\d)$', r'\\1', regex=True)

    print("  > 正在建立 Dummies...")
    dummy_cols = ['IS_SELF_TXN', 'TO_ACCT_TYPE', 'FROM_ACCT_TYPE', 'CHANNEL_TYPE', 'txn_hour', 'currency_type']
    df_dummies = pd.get_dummies(df, columns=dummy_cols, prefix_sep='_', dummy_na=False)

    print("  > 正在定義匯總規則...")
    agg_dict = {
        'TXN_AMT': ['mean', 'std', 'max', 'min'],
        'TXN_DATE': ['max', 'min']
    }
    dummy_col_prefixes = [f'{col}_' for col in dummy_cols]
    dummy_sum_cols = [col for col in df_dummies.columns if any(col.startswith(prefix) for prefix in dummy_col_prefixes)]
    
    for col in dummy_sum_cols:
        if col in df_dummies.columns:
            agg_dict[col] = 'sum'

    print("  > 正在執行分組與匯總...")
    valid_agg_cols = [col for col in agg_dict.keys() if col in df_dummies.columns]
    valid_agg_dict = {col: agg_dict[col] for col in valid_agg_cols}
    
    cols_for_grouping = group_keys + valid_agg_cols
    missing_cols = [col for col in cols_for_grouping if col not in df_dummies.columns]
    if any(col in missing_cols for col in group_keys):
         print(f"  > 錯誤：分組鍵 {group_keys} 不在資料欄位中。")
         return pd.DataFrame()
    
    valid_cols_for_grouping = [col for col in cols_for_grouping if col not in missing_cols]
    
    for col in valid_cols_for_grouping:
        if col not in df_dummies.columns:
             df_dummies[col] = np.nan
             
    df_for_agg = df_dummies[valid_cols_for_grouping].copy()
    
    valid_agg_dict = {k: v for k, v in valid_agg_dict.items() if k in valid_cols_for_grouping}
    
    grouped_df = df_for_agg.groupby(group_keys).agg(valid_agg_dict)
    
    total_counts = df_dummies.groupby(group_keys).size().to_frame('total_txn_count')
    final_df = pd.concat([grouped_df, total_counts], axis=1)

    print("  > 正在清理欄位名稱...")
    final_df.columns = ['_'.join(map(str, col)).strip() if isinstance(col, tuple) else col for col in final_df.columns.values]
    final_df = final_df.rename(columns={
        'TXN_AMT_mean': 'txn_amt_mean', 'TXN_AMT_std': 'txn_amt_std',
        'TXN_AMT_max': 'txn_amt_max', 'TXN_AMT_min': 'txn_amt_min',
        'TXN_DATE_max': 'txn_date_max', 'TXN_DATE_min': 'txn_date_min'
    })
    final_df.columns = [col.replace('_sum', '_count') for col in final_df.columns]
    
    print("  > 正在計算衍生特徵...")
    final_df['txn_amt_std'] = final_df['txn_amt_std'].fillna(0)
    if 'txn_date_max' in final_df.columns and 'txn_date_min' in final_df.columns:
        final_df['txn_date_range'] = final_df['txn_date_max'] - final_df['txn_date_min']
        final_df['txn_frequency'] = final_df['total_txn_count'] / (final_df['txn_date_range'] + 1e-6)
        final_df.loc[final_df['txn_date_range'] == 0, 'txn_frequency'] = final_df['total_txn_count']
    else:
        print("  > 警告: 缺少 TXN_DATE 欄位，無法計算 txn_date_range 和 txn_frequency。")
        final_df['txn_date_range'] = np.nan
        final_df['txn_frequency'] = np.nan

    print("  > 正在計算比例並確保所有欄位存在...")
    all_categories = {
        'IS_SELF_TXN': ['UNK', 'N', 'Y'],
        'TO_ACCT_TYPE': ['01', '02'],
        'FROM_ACCT_TYPE': ['01', '02'],
        'CHANNEL_TYPE': ['1', '2', '3', '4', '5', '6', '7', '99', 'UNK'],
        'txn_hour': [str(h) for h in range(24)] + ['-1'],
        'currency_type': ['AUD', 'CAD', 'CHF', 'CNY', 'EUR', 'GBP', 'HKD', 
                          'JPY', 'NZD', 'SGD', 'THB', 'TWD', 'USD', 'ZAR', 'UNK']
    }
    
    all_output_cols = ['total_txn_count', 'txn_amt_mean', 'txn_amt_std', 'txn_amt_max', 
                       'txn_amt_min', 'txn_date_max', 'txn_date_min', 'txn_date_range', 
                       'txn_frequency']
    
    for prefix, values in all_categories.items():
        for val in values:
            all_output_cols.append(f'{prefix}_{val}_count')
            all_output_cols.append(f'{prefix}_{val}_ratio')

    for prefix, values in all_categories.items():
        for val in values:
            count_col = f'{prefix}_{val}_count'
            ratio_col = f'{prefix}_{val}_ratio'
            
            if count_col not in final_df.columns:
                final_df[count_col] = 0
            
            final_df[ratio_col] = 0.0
            if 'total_txn_count' in final_df.columns:
                mask = final_df['total_txn_count'] > 0
                if mask.any():
                    final_df.loc[mask, ratio_col] = final_df.loc[mask, count_col] / final_df.loc[mask, 'total_txn_count']
            else:
                 final_df[ratio_col] = 0.0

    final_cols = [col for col in all_output_cols if col in final_df.columns]
    final_df = final_df[final_cols]
    
    print("  > 特徵計算完成。")
    # 函式回傳時，index 仍然是 group_key (acct)
    return final_df

# ------------------------------------------------------------------------
# --- 重寫 process_features_for_modeling 函式以支援多個 FLAG ---
# ------------------------------------------------------------------------
def process_features_for_modeling(input_df, use_flag_splitting=True, group_key='acct'):
    """
    接收一個 DataFrame，而不是檔案路徑
    
    V6 修改:
    - use_flag_splitting=True 時，會動態偵測所有 'FLAG' 欄位中的唯一值
      (例如 'FROM', 'TO', 'FROM_NEXT', 'TO_BEFORE')
    - 並為每個 FLAG 值計算特徵，然後冠上對應的前綴 (e.g., 'FROM_')
    - 最後將所有特徵合併 (merge)
    """
    print(f"\n--- 正在處理傳入的 DataFrame (Flag Open: {use_flag_splitting}) ---")
    
    df_full = input_df.copy() 
    
    if group_key not in df_full.columns:
        print(f"錯誤: 欄位 '{group_key}' (或 'ACCT') 不在 DataFrame 中。")
        return pd.DataFrame()
        
    final_features = None
    
    if use_flag_splitting:
        # --- 【NEW】 動態擴展邏輯 ---
        
        # 1. 檢查 'FLAG' 欄位是否存在
        if 'FLAG' not in df_full.columns:
            print("錯誤: FLAG_OPEN='yes' 但 'FLAG' 欄位不存在於資料中。")
            return pd.DataFrame()

        # 2. 取得所有唯一的 FLAG 值
        unique_flags = df_full['FLAG'].unique()
        print(f"FLAG 模式: 開啟 (Pivoted)。偵測到 {len(unique_flags)} 個 Flags: {unique_flags}")

        all_flag_features = [] # 用來儲存所有處理好的 DataFrame (e.g., features_from, features_to)
        
        # 3. 迭代處理每一個 FLAG
        for flag_value in unique_flags:
            if pd.isna(flag_value):
                print(f"  > 警告: 發現 NaN (空) 的 FLAG 值，將其忽略。")
                continue

            print(f"\n  --- 正在計算 '{flag_value}' 特徵 ---")
            df_flag_subset = df_full[df_full['FLAG'] == flag_value].copy()
            
            if df_flag_subset.empty:
                print(f"  > '{flag_value}' 沒有資料，跳過。")
                continue
            
            # 4. 計算特徵 (_calculate_features 回傳的 DF 是以 acct 為 index)
            features_flag = _calculate_features(df_flag_subset, [group_key])
            
            if features_flag.empty:
                print(f"  > '{flag_value}' 特徵計算結果為空，跳過。")
                continue
            
            # 5. 加上前綴 (e.g., "FROM_")
            prefix = f"{str(flag_value).upper()}_"
            print(f"  > 正在為 '{flag_value}' 特徵加上 '{prefix}' 前綴...")
            features_flag = features_flag.add_prefix(prefix)
            
            # 6. 儲存
            all_flag_features.append(features_flag)

        # 7. 合併所有 Flag 特徵
        if not all_flag_features:
            print("錯誤: 沒有任何 FLAG 特徵被成功計算。")
            return pd.DataFrame()
        
        print(f"\n正在合併所有 {len(all_flag_features)} 個 FLAG 群組的特徵...")
        
        # 使用 reduce 依序將 list 中的 DFs 兩兩合併
        # 合併基準是 DataFrame 的 index (也就是 'acct')
        # how='outer' 確保所有帳戶都會被保留
        final_features = reduce(lambda left, right: pd.merge(
                                    left, 
                                    right, 
                                    left_index=True, 
                                    right_index=True, 
                                    how='outer'
                                ), 
                                all_flag_features)
        
        # 8. 填充 NaN (例如某 acct 只有 FROM 特徵, 沒有 TO 特徵) 並 reset_index
        final_features = final_features.fillna(0).reset_index()
        # --- 【NEW】 邏輯結束 ---

    else:
        # (原有的 'no' 邏輯保持不變)
        print("FLAG 模式: 關閉 (Combined)")
        print("正在計算 'Combined' 特徵...")
        final_features = _calculate_features(df_full, [group_key])
        final_features = final_features.reset_index()

    print(f"--- DataFrame 特徵工程完成 ---")
    return final_features


# ------------------------------------------------------------------------
# --- 3. 主程式：Bagging (Ensemble) + Undersampling ---
# ------------------------------------------------------------------------

def load_data(file_path):
    """ 
    嘗試用 UTF-8 和 Big5 讀取 CSV 
    自動將 'ACCT' -> 'acct'
    自動將 'CURRENCY_TYPE' -> 'currency_type'
    """
    if not os.path.exists(file_path):
        print(f"!!! 錯誤: 必要的輸入檔案 {file_path} 找不到! !!!")
        return None
    try:
        df = pd.read_csv(file_path)
        if 'ACCT' in df.columns:
            df = df.rename(columns={'ACCT': 'acct'})
        if 'CURRENCY_TYPE' in df.columns:
            df = df.rename(columns={'CURRENCY_TYPE': 'currency_type'})
        return df
    except UnicodeDecodeError:
        print(f"  > {file_path}: UTF-8 讀取失敗，嘗試 'big5' 編碼...")
        df = pd.read_csv(file_path, encoding='big5')
        if 'ACCT' in df.columns:
            df = df.rename(columns={'ACCT': 'acct'})
        if 'CURRENCY_TYPE' in df.columns:
            df = df.rename(columns={'CURRENCY_TYPE': 'currency_type'})
        return df
    except Exception as e:
        print(f"讀取檔案 {file_path} 失敗: {e}")
        return None

if __name__ == "__main__":
    
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    print("===== 步驟 1: 設置環境與讀取資料 =====")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"所有輸出將儲存於: {OUTPUT_DIR}")
    print(f"本次執行時間戳記: {timestamp}")

    # ========================================================================
    # --- 【NEW】 (您的需求) ---
    # --- 儲存本次執行的參數 ---
    print(f"正在儲存參數設定檔...")
    try:
        # 1. 建立參數字典
        params_to_log = {
            'run_timestamp': timestamp,
            'FLAG_OPEN': FLAG_OPEN,
            'N_TO_PREDICT_POSITIVE': N_TO_PREDICT_POSITIVE,
            'N_ESTIMATORS_ENSEMBLE': N_ESTIMATORS_ENSEMBLE,
            'OUTPUT_DIR': OUTPUT_DIR
        }
        
        # 2. 將特徵群組開關 (FEATURE_GROUPS_TO_USE) 加入字典
        #    (加上 'FEAT_' 前綴, 方便辨識)
        for key, value in FEATURE_GROUPS_TO_USE.items():
            params_to_log[f'FEAT_{key}'] = value
            
        # 3. 轉換為 DataFrame (一行)
        params_df = pd.DataFrame([params_to_log])
        
        # 4. 定義檔案名稱並儲存
        param_filename = os.path.join(OUTPUT_DIR, f'parameters_{timestamp}.csv')
        params_df.to_csv(param_filename, index=False, encoding='utf-8-sig')
        print(f"參數已儲存至: {param_filename}")
        
    except Exception as e:
        print(f"!!! 警告: 儲存參數時發生錯誤: {e} !!!")
    # ========================================================================

    print(f"參數: FLAG_OPEN={FLAG_OPEN}, Model=Ensemble of {N_ESTIMATORS_ENSEMBLE} RandomForests (Undersampled)")
    print(f"策略: 將預測機率最高的前 {N_TO_PREDICT_POSITIVE} 筆標記為 1")
    
    # 讀取資料
    df_alert_raw = load_data('acct_alert_view_export.csv')
    df_predict_raw = load_data('acct_predict_view_export.csv')
    df_order = load_data('acct_predict.csv') # 這是提交順序檔

    if df_alert_raw is None or df_predict_raw is None or df_order is None:
        print("程式因缺少檔案而中止。")
    else:
        print("\n===== 步驟 2: 特徵工程 (FE) =====")
        
        if 'acct' not in df_alert_raw.columns or 'acct' not in df_predict_raw.columns:
            print("錯誤: 'acct' 欄位不存在於 alert 或 predict 檔案中。")
            
        else:
            print("--- 處理正樣本 (Alerts) ---")
            features_positive = process_features_for_modeling(
                input_df=df_alert_raw,
                use_flag_splitting=(FLAG_OPEN == 'yes')
            )
            features_positive['Label'] = 1
            print(f"正樣本特徵維度: {features_positive.shape}")

            print("\n--- 處理 *所有* 負樣本 (Predicts) ---")
            features_negative_all = process_features_for_modeling(
                input_df=df_predict_raw,
                use_flag_splitting=(FLAG_OPEN == 'yes')
            )
            features_negative_all['Label'] = 0
            print(f"負樣本特徵維度: {features_negative_all.shape}")
            
            num_positive = len(features_positive)
            if num_positive == 0:
                print("!!! 錯誤: 正樣本為 0 筆，無法訓練。 !!!")
            
            else:
                # ----------------------------------------------
                # 步驟 3: (*** V6 核心 ***) 訓練 N 個模型
                # ----------------------------------------------
                print(f"\n===== 步驟 3: 訓練 {N_ESTIMATORS_ENSEMBLE} 個獨立的 RF 模型 =====")
                
                models = [] # 儲存所有訓練好的模型
                all_feature_importances = [] # 儲存所有模型的特徵重要性
                
                
                # ========================================================================
                # --- 【需求 1 v2】特徵群組選取邏輯 ---
                #
                # 1. 取得所有 FE 產生的特徵
                X_positive_full = features_positive.drop(columns=['acct', 'Label'])
                y_positive = features_positive['Label']
                
                X_negative_all_full = features_negative_all.drop(columns=['acct', 'Label'])
                y_negative_all = features_negative_all['Label']
                
                all_possible_features = X_positive_full.columns.tolist()
                
                # 2. 根據 FEATURE_GROUPS_TO_USE 建立要使用的特徵列表
                selected_features = []
                
                # --- 【FIX】 (v6.1) ---
                # 基礎統計特徵 (txn_stats, txn_date_stats, txn_amt_stats)
                # 在 FLAG_OPEN='yes' 時，它們會帶有 'FROM_', 'TO_' 等前綴
                # 我們需要一個更通用的方法來匹配它們
                
                base_stats_groups = {
                    'txn_stats': ['total_txn_count', 'txn_frequency'],
                    'txn_date_stats': ['txn_date_max', 'txn_date_min', 'txn_date_range'],
                    'txn_amt_stats': ['txn_amt_mean', 'txn_amt_std', 'txn_amt_max', 'txn_amt_min']
                }

                for group_key, base_features in base_stats_groups.items():
                    if FEATURE_GROUPS_TO_USE.get(group_key, False):
                        for col in all_possible_features:
                            # 檢查欄位是否以任何一個 base_feature 結尾
                            if any(col.endswith(base_feat) for base_feat in base_features):
                                selected_features.append(col)

                # 分類特徵 (Distribution features)
                dist_groups = {
                    'is_self_txn_dist': 'IS_SELF_TXN_',
                    'to_acct_type_dist': 'TO_ACCT_TYPE_',
                    'from_acct_type_dist': 'FROM_ACCT_TYPE_',
                    'channel_type_dist': 'CHANNEL_TYPE_',
                    'txn_hour_dist': 'txn_hour_',
                    'currency_type_dist': 'currency_type_'
                }

                for group_key, prefix_str in dist_groups.items():
                    if FEATURE_GROUPS_TO_USE.get(group_key, False):
                        for col in all_possible_features:
                            if prefix_str in col:
                                selected_features.append(col)
                # --- 【FIX】 (v6.1) 邏輯結束 ---

                # 3. 篩選出實際存在於 FE 結果中的特徵 (並移除重複)
                feature_names = sorted(list(set(col for col in selected_features if col in all_possible_features)))
                
                if not feature_names:
                    print("  > !!! 警告: 根據設定，沒有選取任何特徵! 模型將無法訓練! !!!")
                else:
                    print(f"  > (特徵選取) 根據群組設定，共選取 {len(feature_names)} / {len(all_possible_features)} 個特徵。")

                # 4. 產生最終用於訓練的 X
                X_positive = X_positive_full[feature_names]
                
                # 5. 確保 X_negative_all 欄位一致且順序相同 (reindex 確保欄位對齊並用 0 填充)
                X_negative_all = X_negative_all_full.reindex(columns=feature_names, fill_value=0)
                # ========================================================================


                for i in range(N_ESTIMATORS_ENSEMBLE):
                    #print(f"--- 正在訓練模型 {i+1}/{N_ESTIMATORS_ENSEMBLE} ---")
                    
                    # 1. 抽樣負樣本
                    X_neg_sample = X_negative_all.sample(n=num_positive, random_state=42 + i)
                    y_neg_sample = y_negative_all.loc[X_neg_sample.index]
                    
                    # 2. 組合為 1:1 訓練集
                    X_train = pd.concat([X_positive, X_neg_sample], ignore_index=True)
                    y_train = pd.concat([y_positive, y_neg_sample], ignore_index=True)
                    
                    # 3. 填充 NaN
                    X_train = X_train.fillna(0)
                    
                    # 4. 建立模型
                    model = RandomForestClassifier(
                        n_estimators=200,      
                        random_state=42 + i,
                        n_jobs=-1,
                        max_depth=20,          
                        min_samples_leaf=5,
                        oob_score=False 
                    )
                    
                    # 5. 訓練
                    model.fit(X_train, y_train)
                    
                    # 6. 儲存
                    models.append(model)
                    all_feature_importances.append(model.feature_importances_)
                    #print(f"模型 {i+1} 訓練完成。")

                print(f"\n--- N個模型全部訓練完成 ---")

                # ----------------------------------------------
                # 步驟 4: 準備 *完整* 預測資料 (X_test)
                # ----------------------------------------------
                print("\n===== 步驟 4: 準備 *完整* 預測資料 (共 {len(features_negative_all)} 筆) =====")
                
                X_test = features_negative_all.drop(columns=['acct', 'Label'])
                X_test_accts = features_negative_all[['acct']]
                
                # 確保欄位對齊
                X_test = X_test.fillna(0) 
                X_test = X_test.reindex(columns=feature_names, fill_value=0) 

                # ----------------------------------------------
                # 步驟 5: 產生預測檔 (融合 N 個模型的機率)
                # ----------------------------------------------
                print("\n===== 步驟 5: 產生預測結果 (融合 N 個模型) =====")
                
                all_probs = []
                print("正在從所有模型獲取機率...")
                for i, model in enumerate(models):
                    #print(f"  > 模型 {i+1} 正在預測...")
                    prob_model_i = model.predict_proba(X_test)[:, 1]
                    all_probs.append(prob_model_i)
                
                # 1. 取得機率 (平均值)
                pred_probs_avg = np.mean(all_probs, axis=0)

                results_df = X_test_accts.copy()
                results_df['probability'] = pred_probs_avg
                
                # 2. 找出機率最高的前 N 筆
                top_n_indices = np.argsort(pred_probs_avg)[-N_TO_PREDICT_POSITIVE:]
                
                # 3. 建立 label 欄位，預設為 0
                results_df['label'] = 0
                
                # 4. 將機率最高的前 N 筆的 'label' 設為 1
                results_df.iloc[top_n_indices, results_df.columns.get_loc('label')] = 1

                print(f"已產生融合機率，並將機率最高的前 {N_TO_PREDICT_POSITIVE} 筆標記為 1。")
                print("預測標籤分佈:")
                print(results_df['label'].value_counts())

                # ----------------------------------------------
                # 步驟 6: 排序並儲存 CSV
                # ----------------------------------------------
                print("\n===== 步驟 6: 排序並儲存提交檔案 =====")
                
                if 'acct' not in df_order.columns:
                    if len(df_order.columns) > 0:
                        original_col = df_order.columns[0]
                        df_order = df_order.rename(columns={original_col: 'acct'})
                        print(f"警告: 已將 {original_col} 視為 'acct' 欄位。")
                
                final_submission = df_order[['acct']].merge(results_df, on='acct', how='left')
                
                final_submission['probability'] = final_submission['probability'].fillna(0)
                final_submission['label'] = final_submission['label'].fillna(0).astype(int)

                # ========================================================================
                # --- 【需求 2】分拆提交檔案 (保留) ---
                
                # 1. 儲存 "平台" 提交檔案 (acct, label)
                submission_platform_df = final_submission[['acct', 'label']]
                submission_platform_filename = os.path.join(OUTPUT_DIR, f'submission_platform_{timestamp}.csv')
                submission_platform_df.to_csv(submission_platform_filename, index=False, encoding='utf-8-sig')
                print(f"平台提交檔案 (acct, label) 已儲存: {submission_platform_filename}")
                
                # 2. 儲存 "機率" 檔案 (acct, probability) - 用於本地分析或融合
                submission_proba_df = final_submission[['acct', 'probability']]
                submission_proba_filename = os.path.join(OUTPUT_DIR, f'submission_probabilities_{timestamp}.csv')
                submission_proba_df.to_csv(submission_proba_filename, index=False, encoding='utf-8-sig')
                print(f"機率檔案 (acct, probability) 已儲存: {submission_proba_filename}")
                # ========================================================================

                print("\n提交檔案 (前5筆, 包含機率與標籤):")
                print(final_submission.head())
                print("\n提交檔案 (依機率排序，看最高的):")
                print(final_submission.sort_values(by='probability', ascending=False).head())

                # ----------------------------------------------
                # 步驟 7: 儲存特徵重要性
                # ----------------------------------------------
                print("\n===== 步驟 7: 儲存特徵重要性 (融合 N 個模型) =====")
                
                try:
                    # 計算平均重要性
                    avg_fi = np.mean(all_feature_importances, axis=0)
                    
                    fi_df = pd.DataFrame(
                        {'feature': feature_names, 'importance_mean': avg_fi}
                    )
                    fi_df = fi_df.sort_values(by='importance_mean', ascending=False)

                    feature_importance_filename = os.path.join(OUTPUT_DIR, f'feature_importance_rf_ensemble_{timestamp}.csv')
                    fi_df.to_csv(feature_importance_filename, index=False, encoding='utf-8-sig')
                    print(f"特徵重要性檔案已儲存: {feature_importance_filename}")
                    print("特徵重要性 (Top 15):")
                    print(fi_df.head(15))
                    
                except Exception as e:
                    print(f"計算或儲存特徵重要性時發生錯誤: {e}")

            print("\n===== 所有處理已完成 (使用 Ensemble + Undersampling + Top N) =====")
