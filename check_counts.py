import pandas as pd

file_path = r"c:\agent_b\data\public\1.병원정보서비스(2025.12.).xlsx"
df = pd.read_excel(file_path)

target_types = ['종합병원', '상급종합']
df_filtered = df[df['종별코드명'].isin(target_types)].copy()

print("Hospital counts per city/province (시도코드명):")
print(df_filtered['시도코드명'].value_counts())
