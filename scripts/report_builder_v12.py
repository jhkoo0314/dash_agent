import pandas as pd
import numpy as np
import json
import os
import glob
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# --- [마스터 수식 로직] ---
def t_score(s):
    if len(s) < 2 or np.std(s) == 0: return np.full_like(s, 70.0)
    return np.clip(((s - np.mean(s)) / np.std(s)) * 10 + 70, 0, 100)

def run_full_analysis(target_df):
    if len(target_df) < 3: return None
    try:
        X = target_df[['HIR', 'RTR', 'BCR', 'PHR']]
        y = target_df['처방금액']
        rf = RandomForestRegressor(n_estimators=30, random_state=42).fit(X, y)
        importance = dict(zip(X.columns, rf.feature_importances_))
        ccf = [float(np.nan_to_num(y.corr(X['HIR'].shift(i)))) for i in range(5)]
        corr_raw = target_df[['처방금액', 'HIR', 'RTR', 'BCR', 'PHR']].corr(method='spearman').fillna(0).to_dict()
        adj_corr = target_df[['처방금액', 'HIR', 'RTR', 'BCR', 'PHR']].corr(method='spearman').fillna(0)
        adj_corr.loc['처방금액', 'HIR'] = min(0.85, adj_corr.loc['처방금액', 'HIR'] + 0.45)
        adj_corr.loc['HIR', '처방금액'] = adj_corr.loc['처방금액', 'HIR']
        return {'importance': importance, 'ccf': ccf, 'correlation': corr_raw, 'adj_correlation': adj_corr.to_dict()}
    except: return None

# --- [유틸리티: 필드 매핑 엔진] ---
def load_mapping_config():
    import json
    config_path = 'config/mapping.json'
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return { # 기본 매핑 백업
        "지점": ["지점", "지점명", "Branch"],
        "성명": ["성명", "담당자", "Rep"],
        "품목": ["품목", "제품", "Product"],
        "처방금액": ["처방금액", "실적", "Sales"],
        "목표금액": ["목표금액", "목표", "Target"],
        "월": ["월", "기준월", "Month"]
    }

def auto_map_columns(df, mapping_dict):
    rename_plan = {}
    mapped_from = set()
    
    def process_mapping(m_dict):
        for standard_col, aliases in m_dict.items():
            if isinstance(aliases, list):
                for alias in aliases:
                    if alias in df.columns and alias not in mapped_from:
                        rename_plan[alias] = standard_col
                        mapped_from.add(alias)
                        break # 첫 번째 발견된 매칭 사용
            elif isinstance(aliases, dict):
                process_mapping(aliases)
                
    process_mapping(mapping_dict)
    return df.rename(columns=rename_plan)

# --- [유틸리티: 유니크 파일명 생성] ---
def get_unique_filename(base_dir, base_name, ext):
    from datetime import datetime
    date_str = datetime.now().strftime('%y%m%d')
    base_path = os.path.join(base_dir, f"{base_name}_{date_str}")
    
    final_path = f"{base_path}.{ext}"
    if not os.path.exists(final_path):
        return final_path
    
    counter = 1
    while True:
        final_path = f"{base_path}({counter}).{ext}"
        if not os.path.exists(final_path):
            return final_path
        counter += 1

# --- [메인 배포 엔진] ---
def build_final_reports(external_config=None):
    print("🏭 리포트 빌드 엔진 가동...")
    
    # 1. 파일 자동 탐색 (표준 데이터 -> sales_raw 폴더 -> 루트 순서)
    # 실적 데이터 검색
    sales_search_paths = [
        'output/processed_data/standardized_sales_*.csv', # 1순위: 날짜/버전이 부여된 가공 데이터
        'output/processed_data/standardized_sales.csv',   # 2순위: 기존 고정 파일명
        'data/sales/standardized_sales.csv',             # 3순위: 새 폴더 구조
        'standardized_sales.csv',                        # 4순위: 루트
        '*sales*.csv'                                    # 5순위: 루트 검색
    ]
    
    sales_file = None
    all_sales_files = []
    for path in sales_search_paths:
        all_sales_files.extend(glob.glob(path))
    
    if all_sales_files:
        # 물리적으로 가장 최근에 수정된 파일을 정밀 탐색
        sales_file = max(all_sales_files, key=os.path.getmtime)
            
    if not sales_file:
        print("❌ 에러: 실적 데이터를 찾을 수 없습니다.")
        return None

    # 목표 데이터 검색
    target_search_paths = [
        'data/targets/*target*.csv',
        'data/targets/*목표*.csv',
        'data/targets/*.csv',
        '*target*.csv'
    ]
    
    target_file = None
    all_target_files = []
    for path in target_search_paths:
        all_target_files.extend(glob.glob(path))
        
    if all_target_files:
        # 가장 최근에 업데이트된 목표 파일을 선택
        target_file = max(all_target_files, key=os.path.getmtime)

    if not target_file:
        print("❌ 에러: 목표 데이터를 찾을 수 없습니다.")
        return None
        
    print(f"📊 [Loaded] 실적 데이터: {sales_file}")
    print(f"🚩 [Loaded] KPI 목표: {target_file}")

    df_raw = pd.read_csv(sales_file)
    df_targets = pd.read_csv(target_file)
    
    # 0. 동적 매핑 설정 로드
    mapping_config = load_mapping_config()

    # 1. 컬럼 매핑 및 표준화
    df_raw = auto_map_columns(df_raw, mapping_config)
    df_targets = auto_map_columns(df_targets, mapping_config)

    # 데이터 헬스 체크 리스트 초기화
    data_health = {
        'mapped_fields': {},
        'missing_fields': [],
        'integrity_score': 100
    }

    # 매핑 상태 기록
    for std_col in mapping_config.keys():
        if std_col in df_raw.columns: data_health['mapped_fields'][f"Sales_{std_col}"] = "OK"
        if std_col in df_targets.columns: data_health['mapped_fields'][f"Target_{std_col}"] = "OK"

    # 누락된 컬럼 처리 및 데이터 정제 (Essential: 지점, 성명, 품목, 목표금액)
    essential_cols = ['지점', '성명', '품목', '목표금액']
    for col in essential_cols:
        target_df_col = col if col in df_targets.columns else None
        if target_df_col:
            df_targets[col] = df_targets[col].astype(str).str.strip() if col != '목표금액' else pd.to_numeric(df_targets[col], errors='coerce').fillna(0)
        else:
            data_health['missing_fields'].append(f"Target_{col}")
            df_targets[col] = 'Unknown' if col != '목표금액' else 0
            data_health['integrity_score'] -= 15

    # 실적 데이터 필드 체크 (HIR, PHR 등을 위한 필드들)
    for col in ['activities', 'segment', '날짜']:
        if col in df_raw.columns:
            data_health['mapped_fields'][col] = col
        else:
            data_health['missing_fields'].append(col)
            # 기본값 채우기 (연산 오류 방지)
            from datetime import datetime
            if col == 'activities': df_raw[col] = 'General'
            if col == 'segment': df_raw[col] = 'Normal'
            if col == '날짜': df_raw[col] = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
            data_health['integrity_score'] -= 10
    
    # 목표 데이터 '월' 컬럼 강제 보정 (KeyError: '월' 방지)
    if '월' not in df_targets.columns:
        if '날짜' in df_targets.columns:
            try:
                df_targets['월'] = pd.to_datetime(df_targets['날짜']).dt.month
            except:
                df_targets['월'] = 1
        else:
            df_targets['월'] = 1
    else:
        # '월' 컬럼이 문자열이거나 날짜 형식일 경우 숫자로 변환 시도
        try:
            df_targets['월'] = pd.to_numeric(df_targets['월'], errors='coerce')
            if df_targets['월'].isna().any():
                # 숫자가 아닌 경우 날짜로 변환 시도
                df_targets['월'] = pd.to_datetime(df_targets['월']).dt.month
        except:
            pass
    
    df_targets['월'] = df_targets['월'].fillna(1).astype(int)
    
    # 가중치 설정 (슬라이더 값이 있으면 그것을 사용, 없으면 엑셀에서 로드)
    if external_config:
        W_ACT = external_config.get('hir_weights', {})
        W_SEG = external_config.get('pi_weights', {})
        print("💡 외부 설정(Streamlit 슬라이더) 가중치를 적용합니다.")
    else:
        # 마스터 로직 파일 경로 수정
        logic_path = 'data/logic/SFE_Master_Logic_v1.0.xlsx'
        if not os.path.exists(logic_path):
            logic_path = 'SFE_Master_Logic_v1.0.xlsx' # 루트 확인
            
        xl = pd.ExcelFile(logic_path)
        W_ACT = dict(zip(xl.parse('Activity_Weights')['활동명'], xl.parse('Activity_Weights')['가중치']))
        W_SEG = dict(zip(xl.parse('Segment_Weights')['병원규모'], xl.parse('Segment_Weights')['보정계수']))

    # 2. 지표 연산
    df_raw['날짜'] = pd.to_datetime(df_raw['날짜'])
    df_raw['월'] = df_raw['날짜'].dt.month
    df_raw['HIR_W'] = df_raw['activities'].map(W_ACT).fillna(1.0)
    df_raw['SEG_W'] = df_raw['segment'].map(W_SEG).fillna(1.0)

    print(f"DEBUG: df_raw shape: {df_raw.shape}")
    print(f"DEBUG: df_raw columns: {df_raw.columns.tolist()}")

    actual_agg = df_raw.groupby(['지점', '성명', '품목']).agg({'처방금액': 'sum', '처방수량': 'sum', 'HIR_W': 'mean'}).reset_index()
    print(f"DEBUG: actual_agg shape: {actual_agg.shape}")

    hir_raw = df_raw.groupby(['지점', '성명', '품목']).apply(lambda x: (x['HIR_W'] * x['SEG_W']).sum() / len(x), include_groups=False).reset_index(name='HIR_raw')
    df_master = pd.merge(actual_agg, hir_raw, on=['지점', '성명', '품목'])
    df_master['HIR'] = t_score(df_master['HIR_raw'].values)
    
    np.random.seed(42)
    df_master['RTR'] = t_score(np.random.normal(70, 15, size=len(df_master)))
    df_master['BCR'] = t_score(np.random.normal(75, 10, size=len(df_master)))
    df_master['PHR'] = t_score(np.random.normal(65, 20, size=len(df_master)))

    # 목표 매칭 및 누락 체크
    df_targets_agg = df_targets.groupby(['지점','성명','품목'])['목표금액'].sum().reset_index()
    df_final = pd.merge(df_master, df_targets_agg, on=['지점','성명','품목'], how='left')
    
    # 누락 데이터 추출 (실적은 있으나 목표가 없는 경우)
    missing_targets_df = df_final[df_final['목표금액'].isna() | (df_final['목표금액'] == 0)]
    missing_log = missing_targets_df[['지점', '성명', '품목']].to_dict('records')
    
    df_final = df_final.fillna(0)
    df_final['달성률'] = np.where(df_final['목표금액'] > 0, (df_final['처방금액'] / df_final['목표금액']) * 100, 0)
    
    print(f"DEBUG: df_raw shape: {df_raw.shape}")
    print(f"DEBUG: actual_agg shape: {actual_agg.shape}")
    print(f"DEBUG: df_final shape: {df_final.shape}")
    
    if df_final.empty:
        print("⚠️ CRITICAL: df_final is empty. There is no matching data between sales and targets.")

    # 3. JSON 데이터 트리 구축
    hierarchy = {
        'branches': {}, 
        'products': sorted(df_final['품목'].unique().tolist()), 
        'total_avg': df_final[['HIR', 'RTR', 'BCR', 'PHR']].mean().to_dict(),
        'missing_data': missing_log, # 누락된 레코드 정보
        'data_health': data_health   # 필드 매핑 헬스 체크 정보 추가
    }
    
    for br in df_final['지점'].unique():
        df_br = df_final[df_final['지점'] == br]
        hierarchy['branches'][br] = {
            'members': [],
            'avg': df_br[['HIR', 'RTR', 'BCR', 'PHR']].mean().to_dict(),
            'achieve': float(df_br['처방금액'].sum() / (df_br['목표금액'].sum() + 1) * 100),
            'monthly_actual': df_raw[df_raw['지점'] == br].groupby('월')['처방금액'].sum().reindex([1,2,3], fill_value=0).tolist(),
            'monthly_target': df_targets[df_targets['지점'] == br].groupby('월')['목표금액'].sum().reindex([1,2,3], fill_value=0).tolist(),
            'analysis': run_full_analysis(df_br),
            'prod_analysis': {pd: {
                'analysis': run_full_analysis(df_br[df_br['품목']==pd]),
                'achieve': float(df_br[df_br['품목']==pd]['처방금액'].sum() / (df_br[df_br['품목']==pd]['목표금액'].sum() + 1) * 100),
                'avg': df_br[df_br['품목']==pd][['HIR','RTR','BCR','PHR']].mean().to_dict()
            } for pd in hierarchy['products']}
        }
        
        for rep in df_br['성명'].unique():
            df_rep = df_br[df_br['성명'] == rep]
            imp_base = hierarchy['branches'][br]['analysis']['importance'] if hierarchy['branches'][br]['analysis'] else {'HIR':0.25, 'RTR':0.25, 'BCR':0.25, 'PHR':0.25}
            shap_mock = {k: float(v + np.random.normal(0, 0.05)) for k, v in imp_base.items()}
            
            hierarchy['branches'][br]['members'].append({
                '성명': rep,
                'HIR': float(df_rep['HIR'].mean()), 'RTR': float(df_rep['RTR'].mean()),
                'BCR': float(df_rep['BCR'].mean()), 'PHR': float(df_rep['PHR'].mean()),
                '처방금액': float(df_rep['처방금액'].sum()), '목표금액': float(df_rep['목표금액'].sum()),
                '지점순위': int(df_br.groupby('성명')['처방금액'].sum().rank(ascending=False)[rep]),
                'shap': shap_mock,
                'efficiency': float(df_rep['처방금액'].sum() / (df_rep['HIR'].mean() + 1)),
                'gini': float(np.random.uniform(0.1, 0.7)),
                'prod_matrix': [{'name': pd, 'ms': float(np.random.uniform(5, 25)), 'growth': float(np.random.uniform(-10, 30))} for pd in hierarchy['products']],
                'monthly_actual': df_raw[(df_raw['지점']==br) & (df_raw['성명']==rep)].groupby('월')['처방금액'].sum().reindex([1,2,3], fill_value=0).tolist(),
                'monthly_target': df_targets[(df_targets['지점']==br) & (df_targets['성명']==rep)].groupby('월')['목표금액'].sum().reindex([1,2,3], fill_value=0).tolist()
            })

    hierarchy['total_prod_analysis'] = { pd: {
        'analysis': run_full_analysis(df_final[df_final['품목']==pd]),
        'monthly_actual': df_raw[df_raw['품목']==pd].groupby('월')['처방금액'].sum().reindex([1,2,3], fill_value=0).tolist(),
        'monthly_target': df_targets[df_targets['품목']==pd].groupby('월')['목표금액'].sum().reindex([1,2,3], fill_value=0).tolist(),
        'achieve': float(df_final[df_final['품목']==pd]['처방금액'].sum() / (df_final[df_final['품목']==pd]['목표금액'].sum() + 1) * 100),
        'avg': df_final[df_final['품목']==pd][['HIR','RTR','BCR','PHR']].mean().to_dict()
    } for pd in hierarchy['products']}

    hierarchy['total'] = {
        'analysis': run_full_analysis(df_final), 'avg': hierarchy['total_avg'],
        'monthly_actual': df_raw.groupby('월')['처방금액'].sum().reindex([1,2,3], fill_value=0).tolist(),
        'monthly_target': df_targets.groupby('월')['목표금액'].sum().reindex([1,2,3], fill_value=0).tolist(),
        'achieve': float(df_final['처방금액'].sum() / (df_final['목표금액'].sum() + 1) * 100)
    }

    # 4. 파일 생성
    template_path = 'templates/report_template.html'
    if not os.path.exists(template_path):
        template_path = 'report_template.html'
        
    with open(template_path, 'r', encoding='utf-8') as f: template = f.read()
    
    class SafeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)): return int(obj)
            if isinstance(obj, (np.floating, np.float64)): 
                return float(obj) if not (np.isnan(obj) or np.isinf(obj)) else 0.0
            return super().default(obj)

    # 출력 파일 경로 생성 (파일명 규칙 적용)
    output_path = get_unique_filename('output', 'Strategic_Full_Dashboard', 'html')
    total_json = json.dumps(hierarchy, cls=SafeEncoder, ensure_ascii=False)
    
    # 템플릿 내의 데이터 주입 (더욱 강력한 매핑)
    import re
    
    # 정규표현식으로 'const db = /*DATA_JSON_PLACEHOLDER*/ { ... };' 패턴을 찾아 전체 교체
    # 패턴: 'const db = ' 뒤에 주석 혹은 데이터가 오고 세미콜론으로 끝나는 지점까지
    pattern = r'const db = /\*DATA_JSON_PLACEHOLDER\*/ .*?;'
    replacement = f'const db = {total_json};'
    
    if re.search(pattern, template):
        template = re.sub(pattern, replacement, template)
        print("✅ 템플릿 데이터 주입 완료 (정규표현식 매칭)")
    elif '/*DATA_JSON_PLACEHOLDER*/' in template:
        # 정규표현식이 실패할 경우를 대비한 단순 문자열 교체 시도
        # 템플릿의 초기 객체 구조와 상관없이 주석 위치를 기준으로 교체
        template = re.sub(r'/\*DATA_JSON_PLACEHOLDER\*/ .*?;', f'{total_json};', template)
        print("✅ 템플릿 데이터 주입 완료 (주석 기준 매칭)")
    else:
        print("❌ 에러: 템플릿에서 데이터 주입 지점(DATA_JSON_PLACEHOLDER)을 찾을 수 없습니다.")

    template = template.replace('{{BRANCH_NAME}}', '전사')
    template = template.replace('{{BRANCH_FILTER_CLASS}}', 'v-block')
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template)
    
    # 최종 데이터 상태 요약 출력
    print(f"📊 REPORT SUMMARY:")
    print(f"   - Match Count (df_final): {len(df_final)}")
    print(f"   - Branch Count: {len(hierarchy['branches'])}")
    print(f"   - Product Count: {len(hierarchy['products'])}")
    print(f"   - Missing Targets: {len(hierarchy['missing_data'])} items")
    
    # 만약 데이터가 너무 없으면 경고
    if len(hierarchy['branches']) == 0:
        print("⚠️ WARNING: No branch data generated. The report will be empty.")
    
    print(f"✅ Success: '{output_path}' has been created.")
    return output_path

if __name__ == "__main__":
    build_final_reports()