import pandas as pd
import os

sales_path = r'c:\agent_b\data\sales\hospital_performance.xlsx'
targets_path = r'c:\agent_b\data\targets\hospital_monthly_targets.xlsx'

print(f"Checking files: {os.path.exists(sales_path)}, {os.path.exists(targets_path)}")

if os.path.exists(sales_path):
    df_s = pd.read_excel(sales_path)
    # Check for '목표월' or '활동일자'
    col = '목표월' if '목표월' in df_s.columns else '활동일자'
    df_s['Parsed_Month'] = pd.to_datetime(df_s[col], errors='coerce').dt.month
    print("\nSales Sum by Month:")
    print(df_s.groupby('Parsed_Month')['실적금액'].sum())
    print("\nSales Raw Sample (Month column):")
    print(df_s[[col]].head())

if os.path.exists(targets_path):
    df_t = pd.read_excel(targets_path)
    df_t['Parsed_Month'] = pd.to_datetime(df_t['목표월'], errors='coerce').dt.month
    print("\nTargets Sum by Month:")
    print(df_t.groupby('Parsed_Month')['목표금액'].sum())
    print("\nTargets Raw Sample (Month column):")
    print(df_t[['목표월']].head())
