import pandas as pd
import json

xl = pd.ExcelFile('c:/agent_b/data/logic/SFE_Master_Logic_v1.0.xlsx')
data = {
    'Activity_Weights': pd.read_excel(xl, 'Activity_Weights').to_dict('records'),
    'Segment_Weights': pd.read_excel(xl, 'Segment_Weights').to_dict('records'),
    'Coaching_Rules': pd.read_excel(xl, 'Coaching_Rules').to_dict('records'),
    'System_Setup': pd.read_excel(xl, 'System_Setup').to_dict('records')
}

with open('c:/agent_b/data_logic_dump.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print("Logic data dumped to c:/agent_b/data_logic_dump.json")
