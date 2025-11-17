import pandas as pd
import sys

def generate_export(trans_df, source_df, output_filename):
    """
    Args:
        trans_df (pd.DataFrame): 來自 acct_transaction.csv 的數據
        source_df (pd.DataFrame): 來源數據 (acct_predict.csv 或 acct_alert.csv)
        output_filename (str): 要輸出的 CSV 檔案名稱
    """
    print(f"--- Z 正在處理 {output_filename} ---")

    # 1. 模擬 SQL 中的 (select rownum group_p, a.* from ... order by acct)
    source_sorted = source_df.sort_values('acct').reset_index(drop=True)
    source_grouped = source_sorted.copy()
    source_grouped['group_p'] = source_grouped.index + 1
    
    # 只需要 'acct' 和 'group_p' 進行連接
    source_grouped_simple = source_grouped[['acct', 'group_p']]

    # --- 處理 UNION 的四個部分 ---

    # 2. Part 1: 'FROM_NEXT'
    print("處理 'FROM_NEXT'...")
    # 內層子查詢 b:
    sub_b_join1 = pd.merge(trans_df, source_df, left_on='from_acct', right_on='acct', suffixes=('_trans', '_source'))
    sub_b_pairs = sub_b_join1[['from_acct', 'to_acct']].drop_duplicates()
    sub_b_inner = pd.merge(sub_b_pairs, source_grouped_simple, left_on='from_acct', right_on='acct')
    subquery_b = sub_b_inner[['to_acct', 'group_p']]
    
    # 外層 join: a.from_acct = b.to_acct
    # 這裡 trans_df 有 'to_acct', subquery_b 也有 'to_acct' (作為 join key)
    # pandas 會自動產生 to_acct_trans (來自 trans_df) 和 to_acct_sub (來自 subquery_b)
    from_next_join = pd.merge(trans_df, subquery_b, left_on='from_acct', right_on='to_acct', suffixes=('_trans', '_sub'))
    
    # **[修正點 1]**：丟棄 join key 並將 _trans 欄位名稱改回來，以確保欄位對齊
    from_next_df = from_next_join.drop(columns=['to_acct_sub']).rename(columns={'to_acct_trans': 'to_acct'})
    from_next_df['flag_level'] = 'FROM_NEXT'


    # 3. Part 2: 'FROM'
    print("處理 'FROM'...")
    # SQL: where a.from_acct = b.acct
    from_trans = trans_df[trans_df['from_acct'].isin(source_df['acct'])]
    from_df = pd.merge(from_trans, source_grouped_simple, left_on='from_acct', right_on='acct')
    from_df['flag_level'] = 'FROM'
    from_df = from_df.drop(columns=['acct']) # 刪除 merge 用的 'acct'


    # 4. Part 3: 'TO'
    print("處理 'TO'...")
    # SQL: where a.to_acct = b.acct
    to_trans = trans_df[trans_df['to_acct'].isin(source_df['acct'])]
    to_df = pd.merge(to_trans, source_grouped_simple, left_on='to_acct', right_on='acct')
    to_df['flag_level'] = 'TO'
    to_df = to_df.drop(columns=['acct']) # 刪除 merge 用的 'acct'


    # 5. Part 4: 'TO_BEFORE'
    print("處理 'TO_BEFORE'...")
    # 內層子查詢 b:
    sub_b_join1_tb = pd.merge(trans_df, source_df, left_on='to_acct', right_on='acct', suffixes=('_trans', '_source'))
    sub_b_pairs_tb = sub_b_join1_tb[['from_acct', 'to_acct']].drop_duplicates()
    sub_b_inner_tb = pd.merge(sub_b_pairs_tb, source_grouped_simple, left_on='from_acct', right_on='acct')
    subquery_b_tb = sub_b_inner_tb[['to_acct', 'group_p']]

    # 外層 join: a.to_acct = b.to_acct
    # 這裡 'to_acct' 是 join key，pandas 不會產生後綴，欄位名稱保持 'to_acct'
    to_before_join = pd.merge(trans_df, subquery_b_tb, on='to_acct')
    to_before_df = to_before_join.copy()
    to_before_df['flag_level'] = 'TO_BEFORE'


    # 6. 模擬 UNION
    print("合併 (UNION) 四個部分...")
    # all_parts_df 相當於 SQL 中的外層查詢 ( ... ) a
    all_parts_df_raw = pd.concat([from_next_df, from_df, to_df, to_before_df], ignore_index=True)
    
    # **[修正點 2]**：使用 drop_duplicates() 模擬 SQL 的 'UNION' (而非 'UNION ALL')
    all_parts_df = all_parts_df_raw.drop_duplicates().reset_index(drop=True)
    print(f"UNION ALL 產生 {len(all_parts_df_raw)} 行, UNION (去重後) 產生 {len(all_parts_df)} 行")


    # 7. 執行最外層查詢
    # SQL: from (all_parts_df) a, (source_grouped) b where a.group_p = b.group_p
    # source_grouped 包含 'acct', 'group_p' 和其他欄位 ('level' 或 'alert_date')
    # all_parts_df 包含 'acct_transaction' 的所有欄位, 'group_p', 'flag_level'
    # 'acct' 欄位只存在於 source_grouped 中，因此 merge 不會產生衝突
    final_join = pd.merge(all_parts_df, source_grouped, on='group_p')

    # 8. 整理最後的欄位
    print("選取最終欄位並匯出...")
    
    # **[修正點 3]**：'acct' 欄位直接來自 merge，不需 'acct_source'
    final_output = final_join.rename(columns={'flag_level': 'flag'})
    
    # 根據 SQL 查詢選取最終欄位
    output_columns = [
        'flag', 'acct', 'from_acct_type', 'to_acct_type', 'is_self_txn', 
        'txn_amt', 'txn_date', 'txn_time', 'currency_type', 'channel_type'
    ]
    
    # 確保所有欄位都存在
    final_output_df = final_output[output_columns]
    
    # -----------------------------------------------------------------
    # **[修改]**：將所有欄位名稱轉換為大寫 (依照您上次的要求)
    final_output_df.columns = [col.upper() for col in final_output_df.columns]
    # -----------------------------------------------------------------
    
    # 9. 儲存為 CSV
    final_output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"✅ 成功儲存檔案: {output_filename}")


# --- 主程式 ---
if __name__ == "__main__":
    try:
        # 讀取 CSV 檔案
        print("正在讀取 CSV 檔案...")
        trans = pd.read_csv('acct_transaction.csv')
        alert = pd.read_csv('acct_alert.csv')
        predict = pd.read_csv('acct_predict.csv')
        
        # 轉換 txn_amt 欄位類型以確保 drop_duplicates 正常運作 (以防萬一)
        trans['txn_amt'] = pd.to_numeric(trans['txn_amt'])
        
        print("CSV 檔案讀取完畢。")
        
        # 處理第一個 CSV
        generate_export(trans, predict, 'acct_predict_view_export.csv')
        
        print("\n" + "="*30 + "\n")
        
        # 處理第二個 CSV
        generate_export(trans, alert, 'acct_alert_view_export.csv')
        
        print("\n🎉 處理完成！")

    except FileNotFoundError as e:
        print(f"錯誤: 找不到檔案 {e.filename}。")
        print("請確保 'acct_transaction.csv', 'acct_alert.csv', 'acct_predict.csv' 都在同一個資料夾中。")
    except KeyError as e:
        print(f"錯誤: 找不到欄位 {e}。請檢查 CSV 檔案的欄位名稱是否正確。")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")
        sys.exit(1)
