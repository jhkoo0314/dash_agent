import pandas as pd
import numpy as np
import json
import os
import glob
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# --- [마스터 수식 로직] ---
def t_score(s, t_mean=70.0, t_std=10.0):
    if len(s) < 2 or np.std(s) == 0: return np.full_like(s, t_mean)
    return np.clip(((s - np.mean(s)) / np.std(s)) * t_std + t_mean, 0, 100)

def calc_achieve(actual, target):
    return float((actual / target) * 100) if target and target > 0 else 0.0

def calc_gini(x):
    x = np.sort(np.asarray(x))
    if len(x) == 0 or np.sum(x) == 0: return 0.0
    n = len(x)
    return (np.sum((2 * np.arange(1, n + 1) - n - 1) * x)) / (n * np.sum(x))

# 8대 유효 행동 기준 리스트
ATOMIC_BEHAVIORS = ['PT', '시연', '클로징', '니즈환기', '대면', '컨택', '접근', '피드백']

def run_full_analysis(target_df):
    if len(target_df) < 5: return None
    try:
        # X: 8 Atomic Behaviors, Y: 처방금액
        # 만약 데이터프레임에 해당 8대 행동 컬럼이 없다면 0으로 채움
        for b in ATOMIC_BEHAVIORS:
            if b not in target_df.columns:
                target_df[b] = 0.0

        X = target_df[ATOMIC_BEHAVIORS]
        y = target_df['처방금액']
        
        # 값이 전부 0이면 분석 포기
        if X.sum().sum() == 0 or y.sum() == 0:
            return None

        rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
        importance = dict(zip(X.columns, rf.feature_importances_))
        
        # CCF 및 상관관계는 기존 지표(HIR, RTR, BCR, PHR) 유지
        metrics_cols = ['HIR', 'RTR', 'BCR', 'PHR']
        for m in metrics_cols:
            if m not in target_df.columns: target_df[m] = 70.0

        ccf = [float(np.nan_to_num(y.corr(target_df['HIR'].shift(i)))) for i in range(5)]
        corr_raw = target_df[['처방금액'] + metrics_cols].corr(method='spearman').fillna(0).to_dict()
        adj_corr = target_df[['처방금액'] + metrics_cols].corr(method='spearman').fillna(0).to_dict()
        
        return {'importance': importance, 'ccf': ccf, 'correlation': corr_raw, 'adj_correlation': adj_corr}
    except Exception as e: 
        print(f"[WARN] run_full_analysis error: {e}")
        return None

# --- [유틸리티: 필드 매핑 엔진] ---
def load_mapping_config():
    import json
    config_path = 'config/mapping.json'
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return { # 기본 매핑 백업
        "지점": ["지점", "지점명", "Branch"],
        "성명": ["성명", "담당자명", "담당자", "Rep"],
        "품목": ["품목", "품목명", "제품", "Product"],
        "처방금액": ["처방금액", "실적금액", "실적", "Sales"],
        "목표금액": ["목표금액", "목표", "Target"],
        "월": ["월", "목표월", "기준월", "Month"]
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

def list_files(search_paths):
    files = []
    for path in search_paths:
        files.extend(glob.glob(path))
    uniq = []
    seen = set()
    for f in sorted(files, key=os.path.getmtime, reverse=True):
        norm = os.path.normpath(f)
        if norm not in seen:
            seen.add(norm)
            uniq.append(f)
    return uniq

def read_file(path):
    if path.endswith('.xlsx'):
        return pd.read_excel(path)
    return pd.read_csv(path)

def load_many(files, label):
    frames = []
    for path in files:
        try:
            frames.append(read_file(path))
            print(f"[LOAD:{label}] {path}")
        except Exception as e:
            print(f"[WARN:{label}] load failed: {path} ({e})")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def normalize_key_series(s):
    return s.astype(str).str.strip().str.lower()

def choose_best_key_pair(df_left, left_candidates, df_right, right_candidates):
    best = None
    best_overlap = -1
    for l_col in left_candidates:
        if l_col not in df_left.columns:
            continue
        l_set = set(normalize_key_series(df_left[l_col]).dropna().tolist())
        if not l_set:
            continue
        for r_col in right_candidates:
            if r_col not in df_right.columns:
                continue
            r_set = set(normalize_key_series(df_right[r_col]).dropna().tolist())
            if not r_set:
                continue
            overlap = len(l_set & r_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best = (l_col, r_col, overlap)
    return best

# --- [메인 배포 엔진] ---
def build_final_reports(external_config=None):
    print("[INFO] 리포트 빌드 엔진 가동...")
    
    # 1. 파일 자동 탐색 (표준 데이터 -> sales_raw 폴더 -> 루트 순서)
    # 실적 데이터 검색 (샌드박스 병합본 standardized_sales 우선)
    sales_search_paths = [
        'output/processed_data/standardized_sales_*.csv',
        'output/processed_data/standardized_sales.csv',
        'data/sales/standardized_sales.csv',
        'standardized_sales.csv',
    ]
    sales_fallback_paths = [
        'data/sales/*.xlsx',
        'data/sales/*.csv',
        '*sales*.csv',
        '*sales*.xlsx'
    ]
    
    sales_files = list_files(sales_search_paths)
    use_standardized_sales = len(sales_files) > 0
    if use_standardized_sales:
        sales_files = [sales_files[0]]
    if not sales_files:
        sales_files = list_files(sales_fallback_paths)
        use_standardized_sales = False

    if not sales_files:
        print("[ERROR] 실적 데이터를 찾을 수 없습니다.")
        return None

    # 목표 데이터 검색
    target_search_paths = [
        'data/targets/*target*.csv',
        'data/targets/*target*.xlsx',
        'data/targets/*목표*.csv',
        'data/targets/*목표*.xlsx',
        'data/targets/*.csv',
        'data/targets/*.xlsx',
        '*target*.csv',
        '*target*.xlsx'
    ]
    
    target_files = list_files(target_search_paths)
    if not target_files:
        print("[ERROR] 목표 데이터를 찾을 수 없습니다.")
        return None

    crm_search_paths = [
        'data/crm/*.xlsx',
        'data/crm/*.csv',
        '*crm*.xlsx',
        '*crm*.csv'
    ]
    crm_files = [] if use_standardized_sales else list_files(crm_search_paths)

    print(f"[INFO] 실적 파일 수: {len(sales_files)}")
    print(f"[INFO] KPI 목표 파일 수: {len(target_files)}")
    print(f"[INFO] CRM 파일 수: {len(crm_files)}")
    if use_standardized_sales:
        print("[INFO] standardized_sales 병합본을 기준 데이터로 사용합니다.")

    df_raw = load_many(sales_files, 'SALES')
    df_targets = load_many(target_files, 'TARGETS')
    df_crm = load_many(crm_files, 'CRM') if crm_files else pd.DataFrame()
    
    # 0. 동적 매핑 설정 로드
    mapping_config = load_mapping_config()

    # 1. 컬럼 매핑 및 표준화
    df_raw = auto_map_columns(df_raw, mapping_config)
    df_targets = auto_map_columns(df_targets, mapping_config)
    if not df_crm.empty:
        df_crm = auto_map_columns(df_crm, mapping_config)

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
    for col in ['지점', '성명', '품목']:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].astype(str).str.strip()
        if col in df_targets.columns:
            df_targets[col] = df_targets[col].astype(str).str.strip()
        if not df_crm.empty and col in df_crm.columns:
            df_crm[col] = df_crm[col].astype(str).str.strip()

    if '처방금액' in df_raw.columns:
        df_raw['처방금액'] = pd.to_numeric(df_raw['처방금액'], errors='coerce').fillna(0)
    if '목표금액' in df_targets.columns:
        df_targets['목표금액'] = pd.to_numeric(df_targets['목표금액'], errors='coerce').fillna(0)

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
    for col in ['activities', 'segment', '날짜', '처방수량']:
        if col in df_raw.columns:
            data_health['mapped_fields'][col] = col
        else:
            data_health['missing_fields'].append(col)
            # 기본값 채우기 (연산 오류 방지)
            from datetime import datetime
            if col == 'activities': df_raw[col] = 'General'
            if col == 'segment': df_raw[col] = 'Normal'
            if col == '날짜': df_raw[col] = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
            if col == '처방수량':
                if '처방금액' in df_raw.columns:
                    df_raw['처방수량'] = (df_raw['처방금액'] / 1000).astype(int)
                else:
                    df_raw['처방수량'] = 0
            data_health['integrity_score'] -= 10
    
    # ────────────────────────────────────────────────────────────
    # 월(Month) 파싱 – 단일 헬퍼로 통합하여 중복 제거
    # auto_map_columns이 '목표월' → '월'로 이름만 바꾸므로
    # 값이 '2026-01' 문자열인 경우가 있음. 이를 반드시 숫자로 변환.
    # ────────────────────────────────────────────────────────────
    def parse_month_col(df):
        """df 내에서 월(정수 1-12)을 추출해 반환한다."""
        # 우선순위: '월' → '목표월' → '날짜' → '활동일자'
        for src in ['월', '목표월', '날짜', '활동일자']:
            if src not in df.columns:
                continue
            s = df[src]
            # 이미 정수형이면 바로 반환
            if s.dtype in ['int32', 'int64']:
                return s
            # 날짜/문자열 파싱 시도 (예: '2026-01', '2026-01-15' 등)
            parsed = pd.to_datetime(s, errors='coerce').dt.month
            if parsed.notna().sum() > len(df) * 0.5:   # 절반 이상 파싱 성공 시 채택
                return parsed
            # 숫자만 추출 시도 (예: '1', '01', '2026-01' → '2026' → 월 아님,그냥 skip)
            numeric = pd.to_numeric(s, errors='coerce')
            valid = numeric[(numeric >= 1) & (numeric <= 12)]
            if len(valid) > len(df) * 0.5:
                return numeric
        return pd.Series([1] * len(df), index=df.index)

    df_raw['월']     = parse_month_col(df_raw).fillna(1).astype(int)
    df_targets['월'] = parse_month_col(df_targets).fillna(1).astype(int)
    if not df_crm.empty:
        df_crm['월'] = parse_month_col(df_crm).fillna(1).astype(int)

    print(f"DEBUG: Sales month dist  → {df_raw['월'].value_counts().sort_index().to_dict()}")
    print(f"DEBUG: Target month dist → {df_targets['월'].value_counts().sort_index().to_dict()}")
    if not df_crm.empty:
        print(f"DEBUG: CRM month dist    → {df_crm['월'].value_counts().sort_index().to_dict()}")

    # 기본 컬럼 확인 (처방수량 등)
    if '처방수량' not in df_raw.columns:
        df_raw['처방수량'] = (df_raw['처방금액'] / 1000).astype(int) if '처방금액' in df_raw.columns else 0

    # 매칭 키 자동 정합: 이름/ID 중 겹침이 큰 컬럼 조합을 선택
    branch_pair = choose_best_key_pair(df_raw, ['지점', '지점ID', '지점명'], df_targets, ['지점', '지점ID', '지점명'])
    rep_pair = choose_best_key_pair(df_raw, ['성명', '담당자명', '담당자ID'], df_targets, ['성명', '담당자명', '담당자ID'])
    prod_pair = choose_best_key_pair(df_raw, ['품목', '품목명', '품목ID'], df_targets, ['품목', '품목명', '품목ID'])

    if not branch_pair or not rep_pair or not prod_pair:
        print("[WARN] 키 자동 정합 실패: 기본 키(지점/성명/품목)로 병합합니다.")
        df_raw['__k_branch'] = normalize_key_series(df_raw.get('지점', ''))
        df_raw['__k_rep'] = normalize_key_series(df_raw.get('성명', ''))
        df_raw['__k_prod'] = normalize_key_series(df_raw.get('품목', ''))
        df_targets['__k_branch'] = normalize_key_series(df_targets.get('지점', ''))
        df_targets['__k_rep'] = normalize_key_series(df_targets.get('성명', ''))
        df_targets['__k_prod'] = normalize_key_series(df_targets.get('품목', ''))
    else:
        b_l, b_r, b_ov = branch_pair
        r_l, r_r, r_ov = rep_pair
        p_l, p_r, p_ov = prod_pair
        print(f"[INFO] 키 매칭 선택: branch({b_l}<->{b_r}, {b_ov}), rep({r_l}<->{r_r}, {r_ov}), prod({p_l}<->{p_r}, {p_ov})")
        df_raw['__k_branch'] = normalize_key_series(df_raw[b_l])
        df_raw['__k_rep'] = normalize_key_series(df_raw[r_l])
        df_raw['__k_prod'] = normalize_key_series(df_raw[p_l])
        df_targets['__k_branch'] = normalize_key_series(df_targets[b_r])
        df_targets['__k_rep'] = normalize_key_series(df_targets[r_r])
        df_targets['__k_prod'] = normalize_key_series(df_targets[p_r])

    # CRM 활동명을 실적 데이터(activity)로 매핑
    if not df_crm.empty and 'activities' in df_crm.columns:
        for col in ['지점', '성명', '품목']:
            if col not in df_crm.columns:
                df_crm[col] = 'Unknown'
        df_crm['activities'] = df_crm['activities'].astype(str).str.strip()
        df_crm = df_crm[df_crm['activities'].notna() & (df_crm['activities'] != '')].copy()

        weight_col = None
        for c in ['환산콜(Weighted)', '콜수', '가중치']:
            if c in df_crm.columns:
                weight_col = c
                break
        if weight_col:
            df_crm['act_weight'] = pd.to_numeric(df_crm[weight_col], errors='coerce').fillna(1.0)
        else:
            df_crm['act_weight'] = 1.0

        act_keys = ['지점', '성명', '품목', '월']
        crm_activity = (
            df_crm.groupby(act_keys + ['activities'])['act_weight']
            .sum()
            .reset_index()
            .sort_values(act_keys + ['act_weight'], ascending=[True, True, True, True, False])
            .drop_duplicates(subset=act_keys)
            [act_keys + ['activities']]
        )

        if 'activities' not in df_raw.columns:
            df_raw['activities'] = np.nan
        df_raw = df_raw.merge(crm_activity, on=act_keys, how='left', suffixes=('', '_crm'))
        if 'activities_crm' in df_raw.columns:
            df_raw['activities'] = np.where(
                df_raw['activities_crm'].notna() & (df_raw['activities_crm'].astype(str).str.strip() != ''),
                df_raw['activities_crm'],
                df_raw['activities']
            )
            df_raw = df_raw.drop(columns=['activities_crm'])

        mapped_activity_count = int(df_raw['activities'].notna().sum())
        print(f"DEBUG: CRM activity mapped rows → {mapped_activity_count:,}")
        data_health['mapped_fields']['activities'] = "CRM.activities"
        if 'activities' in data_health['missing_fields']:
            data_health['missing_fields'] = [x for x in data_health['missing_fields'] if x != 'activities']
            
    # 가중치 설정 (슬라이더 값이 있으면 그것을 사용, 없으면 엑셀에서 로드)
    if external_config:
        W_ACT = external_config.get('hir_weights', {})
        W_SEG = external_config.get('pi_weights', {})
        print("[INFO] 외부 설정(Streamlit 슬라이더) 가중치를 적용합니다.")
        T_MEAN, T_STD = 70.0, 10.0
    else:
        # 마스터 로직 파일 경로 수정
        logic_path = 'data/logic/SFE_Master_Logic_v1.0.xlsx'
        if not os.path.exists(logic_path):
            logic_path = 'SFE_Master_Logic_v1.0.xlsx' # 루트 확인
            
        xl = pd.ExcelFile(logic_path)
        W_ACT = dict(zip(xl.parse('Activity_Weights')['활동명'], xl.parse('Activity_Weights')['가중치']))
        W_SEG = dict(zip(xl.parse('Segment_Weights')['병원규모'], xl.parse('Segment_Weights')['보정계수']))
        
        try:
            sys_setup = xl.parse('System_Setup')
            T_MEAN = float(sys_setup.loc[sys_setup['설정항목'].str.contains('T-Score 평균', na=False), '설정값'].values[0])
            T_STD = float(sys_setup.loc[sys_setup['설정항목'].str.contains('T-Score 편차', na=False), '설정값'].values[0])
        except Exception as e:
            print(f"[WARN] T-Score 설정 로드 실패: {e}")
            T_MEAN, T_STD = 70.0, 10.0

    # 2. 지표 연산 및 8대 행동 Atomic 파싱
    # W_ACT 딕셔너리를 기반으로 activities 컬럼의 문장을 파싱해서 빈도/가중치 체크
    w_act_map = {str(k).strip(): v for k, v in W_ACT.items()}
    df_raw['activities'] = df_raw['activities'].astype(str).str.strip()
    
    # 각 row (방문/Call) 별로 8대 행동 점수 매핑 (Atomic Split)
    for b in ATOMIC_BEHAVIORS:
        # activities 내에 해당 단어가 포함되어있으면 1.0 (또는 해당 가중치), 아니면 0.0
        df_raw[b] = df_raw['activities'].apply(lambda x: 1.0 if b in x else 0.0)
    
    # 전체 HIR 연산을 위해: 8대 행동 * 가중치의 합
    df_raw['HIR_W'] = 0.0
    for b in ATOMIC_BEHAVIORS:
        weight = float(w_act_map.get(b, 1.0))
        df_raw['HIR_W'] += df_raw[b] * weight
        
    df_raw['SEG_W'] = df_raw['segment'].map(W_SEG).fillna(1.0)

    # RTR: 날짜_ts 감쇠 로직 $exp(-0.035 \times t)$
    df_raw['날짜_ts'] = pd.to_datetime(df_raw['날짜'], errors='coerce')
    current_time = pd.Timestamp.now()
    t_days = (current_time - df_raw['날짜_ts']).dt.days.clip(lower=0)
    df_raw['RTR_raw'] = np.exp(-0.035 * t_days).fillna(0)

    print(f"DEBUG: df_raw shape: {df_raw.shape}")
    print(f"DEBUG: df_raw columns: {df_raw.columns.tolist()}")

    group_cols = ['지점', '성명', '품목', '__k_branch', '__k_rep', '__k_prod']
    
    # 각 그룹별로 Atomic 8 행동 총합 계산
    atomic_agg_dict = {b: 'sum' for b in ATOMIC_BEHAVIORS}
    agg_dict = {'처방금액': 'sum', '처방수량': 'sum', 'HIR_W': 'mean', 'RTR_raw': 'mean'}
    agg_dict.update(atomic_agg_dict)
    
    actual_agg = df_raw.groupby(group_cols).agg(agg_dict).reset_index()
    print(f"DEBUG: actual_agg shape: {actual_agg.shape}")

    # BCR: 방문 간격 표준편차 $\sigma$ 유도. 일관성이 높으면 표준편차 낮음
    df_sorted = df_raw.sort_values(group_cols + ['날짜_ts'])
    df_sorted['interval'] = df_sorted.groupby(group_cols)['날짜_ts'].diff().dt.days
    # 역수로 해서 값이 클수록(규칙적일수록) 좋게 구성 (interval std가 작을수록 좋음)
    # 0 분모 방지 위해 + 1
    bcr_raw = df_sorted.groupby(group_cols)['interval'].apply(lambda x: 1.0 / (np.std(x) + 1.0) if len(x) > 1 else 0).reset_index(name='BCR_raw')

    hir_raw = df_raw.groupby(group_cols).apply(lambda x: (x['HIR_W'] * x['SEG_W']).sum() / len(x) if len(x)>0 else 0, include_groups=False).reset_index(name='HIR_raw')
    df_master = pd.merge(actual_agg, hir_raw, on=group_cols)
    df_master = pd.merge(df_master, bcr_raw, on=group_cols, how='left')
    
    # 마스터 시트에서 가져온 T_MEAN, T_STD 로 가중평균 환산 (T-Score)
    df_master['HIR'] = t_score(df_master['HIR_raw'].values, T_MEAN, T_STD)
    df_master['RTR'] = t_score(df_master['RTR_raw'].values, T_MEAN, T_STD)
    df_master['BCR'] = t_score(df_master['BCR_raw'].values, T_MEAN, T_STD)
    df_master['PHR'] = np.full_like(df_master['HIR'].values, T_MEAN)

    # standardized_sales에 기존 지표가 있으면 우선 사용
    # standardized_sales에 기존 지표가 있으면 우선 사용
    for metric in ['HIR', 'RTR', 'BCR', 'PHR']:
        target_col = None
        if f"{metric}_Raw" in df_raw.columns:
            target_col = f"{metric}_Raw"
        elif metric in df_raw.columns:
            target_col = metric
            
        if target_col:
            metric_df = df_raw[group_cols + [target_col]].copy()
            metric_df[target_col] = pd.to_numeric(metric_df[target_col], errors='coerce')
            metric_agg = metric_df.groupby(group_cols)[target_col].mean().reset_index(name=f'{metric}_src')
            df_master = df_master.merge(metric_agg, on=group_cols, how='left')
            src = df_master[f'{metric}_src']
            if src.notna().sum() > 0:
                # If values are raw (e.g. 0~5), apply t_score. If already scaled (like 0-100), just use them.
                # Usually standard raw values have small stdev
                if (src.std() or 0) > 0:
                    df_master[metric] = t_score(src.fillna(src.mean()).values, T_MEAN, T_STD)
                else:
                    df_master[metric] = np.full_like(src, T_MEAN) # 기본 점수
            df_master = df_master.drop(columns=[f'{metric}_src'])

    # 목표 매칭 및 누락 체크
    df_targets_agg = df_targets.groupby(['__k_branch','__k_rep','__k_prod'])['목표금액'].sum().reset_index()
    df_final = pd.merge(df_master, df_targets_agg, on=['__k_branch','__k_rep','__k_prod'], how='left')
    
    # 누락 데이터 추출 (실적은 있으나 목표가 없는 경우)
    missing_targets_df = df_final[df_final['목표금액'].isna() | (df_final['목표금액'] == 0)]
    missing_log = missing_targets_df[['지점', '성명', '품목']].to_dict('records')
    
    df_final = df_final.fillna(0)
    df_final['달성률'] = np.where(df_final['목표금액'] > 0, (df_final['처방금액'] / df_final['목표금액']) * 100, 0)
    
    print(f"DEBUG: df_raw shape: {df_raw.shape}")
    print(f"DEBUG: actual_agg shape: {actual_agg.shape}")
    print(f"DEBUG: df_final shape: {df_final.shape}")
    
    if df_final.empty:
        print("[CRITICAL] df_final is empty. There is no matching data between sales and targets.")

    # --- [코칭 룰 엔진] ---
    def get_coaching_message(hir, rtr, bcr, ach, th_hir=70, th_rtr=70, th_bcr=70, th_ach=100):
        # 마스터 로직 코칭 룰 
        if ach >= th_ach:
            if hir >= th_hir and rtr >= th_rtr:
                return "The Masterclass", "현재의 높은 활동량과 우수한 관계 유지 능력을 유지하세요. Best Practice 사례로 공유를 권장합니다."
            elif hir < th_hir and rtr >= th_rtr:
                return "The Relationship Builder", "고객과의 관계는 훌륭하나 활동량이 다소 부족합니다. 방문 커버리지를 늘려 파이프라인을 확장하세요."
            elif hir >= th_hir and rtr < th_rtr:
                return "The Volume Driver", "활동량은 우수하나 관계 깊이가 아쉽습니다. 핵심 고객층에 대한 심층적이고 퀄리티 높은 디테일링이 필요합니다."
            else:
                return "The Lucky Star", "데이터상 유효행동과 관계온도가 낮음에도 목표를 달성했습니다. 외부 요인(시장 상황 등)이나 일회성 매출 여부를 점검하세요."
        else:
            if hir >= th_hir and bcr < th_bcr:
                return "The Erratic Sprinter", "활동량은 많으나 방문이 불규칙합니다. 사전 계획(PHR)을 철저히 기획하여 균일하게 방문 일정을 안배하세요."
            elif rtr < th_rtr and bcr >= th_bcr:
                return "The Routine Visitor", "규칙적으로 꾸준히 방문하나 고객과의 관계 온도가 낮습니다. 단순 제품 전달을 넘어선 솔루션 제안(PT/니즈환기) 스킬 교육이 시급합니다."
            elif hir < th_hir and bcr < th_bcr:
                return "The Ghost Hunter", "활동량과 규칙성 모두 저조합니다. 근태 및 일일 활동 계획에 대한 밀착 코칭과 파이프라인 전면 재설계가 필요합니다."
            else:
                return "The Hard Worker", "성실하게 양질의 활동을 수행하고 있으나 성과로 이어지지 않고 있습니다. 타겟팅(Segment)이나 주력 품목(MS) 전략의 재점검이 필요합니다."

    # 3. JSON 데이터 트리 구축
    hierarchy = {
        'branches': {}, 
        'products': sorted(df_final['품목'].unique().tolist()), 
        'total_avg': df_final[['HIR', 'RTR', 'BCR', 'PHR']].mean().to_dict(),
        'missing_data': missing_log, # 누락된 레코드 정보
        'data_health': data_health   # 필드 매핑 헬스 체크 정보 추가
    }
    
    month_axis = list(range(1, 13))

    # 타겟 월 데이터는 매칭 키로 sales 라벨에 매핑한 뒤 사용
    target_monthly = (
        df_targets[['__k_branch', '__k_rep', '__k_prod', '월', '목표금액']]
        .merge(
            df_final[['__k_branch', '__k_rep', '__k_prod', '지점', '성명', '품목']].drop_duplicates(),
            on=['__k_branch', '__k_rep', '__k_prod'],
            how='inner'
        )
    )

    # 2.5 대표(Rep) 레벨 지표 정규화 (변별력 확보)
    # 개별 품목 T-score의 평균을 쓰면 변별력이 사라지므로(평균회귀), Rep 레벨에서 Raw 점수를 다시 T-score화
    df_rep_raw_calc = df_final.groupby(['지점', '성명']).agg({
        'HIR_raw': 'mean',
        'RTR_raw': 'mean',
        'BCR_raw': 'mean',
        '처방금액': 'sum',
        '목표금액': 'sum'
    }).reset_index()
    
    df_rep_raw_calc['REP_HIR'] = t_score(df_rep_raw_calc['HIR_raw'].values, T_MEAN, T_STD)
    df_rep_raw_calc['REP_RTR'] = t_score(df_rep_raw_calc['RTR_raw'].values, T_MEAN, T_STD)
    df_rep_raw_calc['REP_BCR'] = t_score(df_rep_raw_calc['BCR_raw'].values, T_MEAN, T_STD)
    df_rep_raw_calc['REP_ACH'] = (df_rep_raw_calc['처방금액'] / df_rep_raw_calc['목표금액'] * 100).fillna(0)
    
    # 동적 Threshold (중앙값) 계산
    th_hir = float(df_rep_raw_calc['REP_HIR'].median())
    th_rtr = float(df_rep_raw_calc['REP_RTR'].median())
    th_bcr = float(df_rep_raw_calc['REP_BCR'].median())
    # 달성률은 100%를 기준으로 하되, 전반적으로 낮으면 중앙값 사용 고려 가능하나 일단 100 유지 또는 하향 조정
    th_ach = 100.0 if df_rep_raw_calc['REP_ACH'].max() >= 100 else float(df_rep_raw_calc['REP_ACH'].median())
    
    print(f"DEBUG: Coaching Thresholds -> HIR:{th_hir:.1f}, RTR:{th_rtr:.1f}, BCR:{th_bcr:.1f}, ACH:{th_ach:.1f}")

    for br in df_final['지점'].unique():
        df_br = df_final[df_final['지점'] == br]
        hierarchy['branches'][br] = {
            'members': [],
            'avg': df_br[['HIR', 'RTR', 'BCR', 'PHR']].mean().to_dict(),
            'achieve': calc_achieve(df_br['처방금액'].sum(), df_br['목표금액'].sum()),
            'monthly_actual': df_raw[df_raw['지점'] == br].groupby('월')['처방금액'].sum().reindex(month_axis, fill_value=0).tolist(),
            'monthly_target': target_monthly[target_monthly['지점'] == br].groupby('월')['목표금액'].sum().reindex(month_axis, fill_value=0).tolist(),
            'analysis': run_full_analysis(df_br),
            'prod_analysis': {pd: {
                'analysis': run_full_analysis(df_br[df_br['품목']==pd]),
                'achieve': calc_achieve(df_br[df_br['품목']==pd]['처방금액'].sum(), df_br[df_br['품목']==pd]['목표금액'].sum()),
                'avg': df_br[df_br['품목']==pd][['HIR','RTR','BCR','PHR']].mean().to_dict()
            } for pd in hierarchy['products']}
        }
        
        for rep in df_br['성명'].unique():
            df_rep = df_br[df_br['성명'] == rep]
            rep_analysis = run_full_analysis(df_rep)
            if rep_analysis is not None:
                real_shap = {k: float(v) for k, v in rep_analysis['importance'].items()}
            else:
                real_shap = {b: np.nan for b in ATOMIC_BEHAVIORS}
            
            prod_matrix = []
            rep_raw = df_raw[(df_raw['지점'] == br) & (df_raw['성명'] == rep)]
            total_sales = float(rep_raw['처방금액'].sum()) if not rep_raw.empty else 0.0
            if total_sales > 0 and not rep_raw.empty:
                max_m = rep_raw['월'].max()
                prev_m = max_m - 1
                for pd_name in hierarchy['products']:
                    p_data = rep_raw[rep_raw['품목'] == pd_name]
                    p_sales = float(p_data['처방금액'].sum())
                    ms = (p_sales / total_sales * 100) if total_sales else 0.0
                    cm_sales = float(p_data[p_data['월'] == max_m]['처방금액'].sum())
                    pm_sales = float(p_data[p_data['월'] == prev_m]['처방금액'].sum())
                    if pm_sales > 0:
                        growth = ((cm_sales - pm_sales) / pm_sales) * 100
                    else:
                        growth = 100.0 if cm_sales > 0 else 0.0
                    prod_matrix.append({'name': pd_name, 'ms': ms, 'growth': growth})
            else:
                prod_matrix = [{'name': pd_name, 'ms': 0.0, 'growth': 0.0} for pd_name in hierarchy['products']]
            
            # 4분면 전략 가이드라인: 기준선 계산 및 코칭 액션 트리거
            ms_values = [p['ms'] for p in prod_matrix if p['ms'] > 0]
            avg_ms = float(sum(ms_values) / len(ms_values)) if ms_values else 0.0
            
            # 코칭 메시지 연산
            rep_stats = df_rep_raw_calc[df_rep_raw_calc['성명'] == rep].iloc[0]
            rep_hir = float(rep_stats['REP_HIR'])
            rep_rtr = float(rep_stats['REP_RTR'])
            rep_bcr = float(rep_stats['REP_BCR'])
            rep_ach = float(rep_stats['REP_ACH'])
            
            c_name, c_action = get_coaching_message(rep_hir, rep_rtr, rep_bcr, rep_ach, th_hir, th_rtr, th_bcr, th_ach)

            # Dog(Low MS / Low Growth) 또는 Question Mark(Low MS / High Growth) 파악 
            # (단순화를 위해 ms가 평균 미만인 주력/비주력 품목 중 의미 있는 볼륨 추적)
            weak_products = [p['name'] for p in prod_matrix if p['ms'] > 0 and p['ms'] < (avg_ms * 0.7) and p['growth'] < 0]
            if weak_products:
                c_action += f" (🚨 주의: {', '.join(weak_products)} 품목이 Dog 영역에 위치해 있습니다. 품목 성장이 저조하므로 타겟팅 전략 재수립이 필요합니다.)"

            hierarchy['branches'][br]['members'].append({
                '성명': rep,
                'HIR': rep_hir, 'RTR': rep_rtr, 'BCR': rep_bcr, 'PHR': float(df_rep['PHR'].mean()),
                '처방금액': float(df_rep['처방금액'].sum()), '목표금액': float(df_rep['목표금액'].sum()),
                '지점순위': int(df_br.groupby('성명')['처방금액'].sum().rank(ascending=False)[rep]),
                'shap': real_shap,
                'coach_scenario': c_name,
                'coach_action': c_action,
                'efficiency': float(df_rep['처방금액'].sum() / (rep_hir + 1)),
                'gini': float(calc_gini(df_rep['처방금액'])),
                'avg_ms': avg_ms,
                'prod_matrix': prod_matrix,
                'monthly_actual': df_raw[(df_raw['지점']==br) & (df_raw['성명']==rep)].groupby('월')['처방금액'].sum().reindex(month_axis, fill_value=0).tolist(),
                'monthly_target': target_monthly[(target_monthly['지점']==br) & (target_monthly['성명']==rep)].groupby('월')['목표금액'].sum().reindex(month_axis, fill_value=0).tolist()
            })

    hierarchy['total_prod_analysis'] = { pd: {
        'analysis': run_full_analysis(df_final[df_final['품목']==pd]),
        'monthly_actual': df_raw[df_raw['품목']==pd].groupby('월')['처방금액'].sum().reindex(month_axis, fill_value=0).tolist(),
        'monthly_target': target_monthly[target_monthly['품목']==pd].groupby('월')['목표금액'].sum().reindex(month_axis, fill_value=0).tolist(),
        'achieve': calc_achieve(df_final[df_final['품목']==pd]['처방금액'].sum(), df_final[df_final['품목']==pd]['목표금액'].sum()),
        'avg': df_final[df_final['품목']==pd][['HIR','RTR','BCR','PHR']].mean().to_dict()
    } for pd in hierarchy['products']}

    hierarchy['total'] = {
        'analysis': run_full_analysis(df_final), 'avg': hierarchy['total_avg'],
        'monthly_actual': df_raw.groupby('월')['처방금액'].sum().reindex(month_axis, fill_value=0).tolist(),
        'monthly_target': target_monthly.groupby('월')['목표금액'].sum().reindex(month_axis, fill_value=0).tolist(),
        'achieve': calc_achieve(df_final['처방금액'].sum(), df_final['목표금액'].sum())
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
        print("[INFO] 템플릿 데이터 주입 완료 (정규표현식 매칭)")
    elif '/*DATA_JSON_PLACEHOLDER*/' in template:
        # 정규표현식이 실패할 경우를 대비한 단순 문자열 교체 시도
        # 템플릿의 초기 객체 구조와 상관없이 주석 위치를 기준으로 교체
        template = re.sub(r'/\*DATA_JSON_PLACEHOLDER\*/ .*?;', f'{total_json};', template)
        print("[INFO] 템플릿 데이터 주입 완료 (주석 기준 매칭)")
    else:
        print("[ERROR] 템플릿에서 데이터 주입 지점(DATA_JSON_PLACEHOLDER)을 찾을 수 없습니다.")

    template = template.replace('{{BRANCH_NAME}}', '전사')
    template = template.replace('{{BRANCH_FILTER_CLASS}}', 'v-block')
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template)
    
    # 최종 데이터 상태 요약 출력
    print("[INFO] REPORT SUMMARY:")
    print(f"   - Match Count (df_final): {len(df_final)}")
    print(f"   - Branch Count: {len(hierarchy['branches'])}")
    print(f"   - Product Count: {len(hierarchy['products'])}")
    print(f"   - Missing Targets: {len(hierarchy['missing_data'])} items")
    
    # 만약 데이터가 너무 없으면 경고
    if len(hierarchy['branches']) == 0:
        print("[WARN] No branch data generated. The report will be empty.")
    
    print(f"[OK] '{output_path}' has been created.")
    return output_path

if __name__ == "__main__":
    build_final_reports()
