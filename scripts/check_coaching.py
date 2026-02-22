import re
import json

path = r'c:\agent_b\output\Strategic_Full_Dashboard_260222(29).html'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Look for db assignment
match = re.search(r'const db = (\{.*?\});', content, re.DOTALL)
if match:
    db = json.loads(match.group(1))
    
    scenarios = []
    actions = []
    achs = []
    hirs = []
    
    for br_name, br_data in db['branches'].items():
        for m in br_data['members']:
            scenarios.append(m.get('coach_scenario'))
            actions.append(m.get('coach_action'))
            # Calculate Ach from 처방금액 and 목표금액
            actual = m.get('처방금액', 0)
            target = m.get('목표금액', 0)
            achs.append((actual / target * 100) if target > 0 else 0)
            hirs.append(m.get('HIR', 0))
            
    from collections import Counter
    print("Scenario Distribution:")
    dist = Counter(scenarios)
    for k, v in dist.items():
        print(f"  {k}: {v}")
    
    print(f"\nAch stats: min={min(achs):.1f}, max={max(achs):.1f}, mean={sum(achs)/len(achs):.1f}")
    print(f"HIR stats: min={min(hirs):.1f}, max={max(hirs):.1f}, mean={sum(hirs)/len(hirs):.1f}")
    
    print("\nSamples (first 10 scenarios):")
    for i in range(10):
        print(f"Rep {i}: {scenarios[i]}")
    print("\nAction Example (first 3):")
    for a in actions[:3]:
        print(f"- {a}")
else:
    print("Could not find db in HTML")
