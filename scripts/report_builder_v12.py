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

def calc_gap(actual, target):
    gap_amount = float(actual - target)
    gap_rate = calc_achieve(actual, target) - 100.0 if target and target > 0 else 0.0
    return gap_amount, gap_rate

def calc_gini(x):
    x = np.sort(np.asarray(x))
    if len(x) == 0 or np.sum(x) == 0: return 0.0
    n = len(x)
    return (np.sum((2 * np.arange(1, n + 1) - n - 1) * x)) / (n * np.sum(x))

# 8대 유효 행동 기준 리스트
ATOMIC_BEHAVIORS = ['PT', '시연', '클로징', '니즈환기', '대면', '컨택', '접근', '피드백']
MATRIX_METRICS = ['HIR', 'RTR', 'BCR', 'PHR']

def _zero_corr_dict():
    keys = ['처방금액'] + MATRIX_METRICS
    out = {}
    for r in keys:
        out[r] = {}
        for c in keys:
            out[r][c] = 1.0 if r == c else 0.0
    return out

def _safe_spearman(df_like):
    cols = ['처방금액'] + MATRIX_METRICS
    if df_like is None or len(df_like) < 2:
        return _zero_corr_dict()
    work = df_like.copy()
    for c in cols:
        if c not in work.columns:
            work[c] = 0.0
    work = work[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    if work.shape[0] < 2:
        return _zero_corr_dict()
    corr = work.corr(method='spearman').fillna(0.0)
    for c in cols:
        if c not in corr.index:
            corr.loc[c] = 0.0
        if c not in corr.columns:
            corr[c] = 0.0
    corr = corr.loc[cols, cols]
    for c in cols:
        corr.loc[c, c] = 1.0
    return corr.to_dict()

def build_period_matrices(target_df):
    """월/분기별 실시간 상관관계 + 1개월 후행(4주 보정) 상관관계."""
    empty_month = [_zero_corr_dict() for _ in range(12)]
    empty_quarter = [_zero_corr_dict() for _ in range(4)]
    if target_df is None or len(target_df) == 0:
        return {
            'monthly_correlation': empty_month,
            'monthly_adj_correlation': empty_month,
            'quarterly_correlation': empty_quarter,
            'quarterly_adj_correlation': empty_quarter,
        }

    df = target_df.copy()
    for c in ['월', '처방금액'] + MATRIX_METRICS:
        if c not in df.columns:
            df[c] = 0.0
    df['월'] = pd.to_numeric(df['월'], errors='coerce').fillna(0).astype(int)
    df = df[df['월'].between(1, 12)]
    if df.empty:
        return {
            'monthly_correlation': empty_month,
            'monthly_adj_correlation': empty_month,
            'quarterly_correlation': empty_quarter,
            'quarterly_adj_correlation': empty_quarter,
        }
    group_keys = [k for k in ['월', '__k_branch', '__k_rep', '__k_prod'] if k in df.columns]
    if not group_keys:
        group_keys = ['월']
    agg_ops = {'처방금액': 'sum', 'HIR': 'mean', 'RTR': 'mean', 'BCR': 'mean', 'PHR': 'mean'}
    agg = df[group_keys + ['처방금액', 'HIR', 'RTR', 'BCR', 'PHR']].groupby(group_keys, as_index=False).agg(agg_ops)

    id_keys = [k for k in ['__k_branch', '__k_rep', '__k_prod'] if k in agg.columns]
    prev = agg[['월'] + id_keys + MATRIX_METRICS].copy()
    prev['월'] = prev['월'] + 1
    prev = prev.rename(columns={m: f'{m}_prev' for m in MATRIX_METRICS})
    lagged = agg[['월'] + id_keys + ['처방금액']].merge(prev, on=['월'] + id_keys, how='left')

    monthly_raw = []
    monthly_adj = []
    for m in range(1, 13):
        raw_m = agg[agg['월'] == m][['처방금액'] + MATRIX_METRICS]
        monthly_raw.append(_safe_spearman(raw_m))

        adj_cols = ['처방금액'] + [f'{x}_prev' for x in MATRIX_METRICS]
        adj_m = lagged[lagged['월'] == m][adj_cols].rename(
            columns={f'{x}_prev': x for x in MATRIX_METRICS}
        )
        monthly_adj.append(_safe_spearman(adj_m))

    quarterly_raw = []
    quarterly_adj = []
    for q in range(4):
        months = [q * 3 + 1, q * 3 + 2, q * 3 + 3]
        raw_q = agg[agg['월'].isin(months)][['처방금액'] + MATRIX_METRICS]
        quarterly_raw.append(_safe_spearman(raw_q))

        adj_cols = ['처방금액'] + [f'{x}_prev' for x in MATRIX_METRICS]
        adj_q = lagged[lagged['월'].isin(months)][adj_cols].rename(
            columns={f'{x}_prev': x for x in MATRIX_METRICS}
        )
        quarterly_adj.append(_safe_spearman(adj_q))

    return {
        'monthly_correlation': monthly_raw,
        'monthly_adj_correlation': monthly_adj,
        'quarterly_correlation': quarterly_raw,
        'quarterly_adj_correlation': quarterly_adj,
    }

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
        
        period_mats = build_period_matrices(target_df)
        return {
            'importance': importance,
            'ccf': ccf,
            'correlation': corr_raw,
            'adj_correlation': adj_corr,
            **period_mats
        }
    except Exception as e: 
        print(f"[WARN] run_full_analysis error: {e}")
        return None

def estimate_atomic_importance(df_slice):
    """Fallback atomic importance when model fit is unstable."""
    if df_slice is None or len(df_slice) == 0:
        return {b: 0.0 for b in ATOMIC_BEHAVIORS}
    work = df_slice.copy()
    for b in ATOMIC_BEHAVIORS:
        if b not in work.columns:
            work[b] = 0.0
    if '처방금액' not in work.columns:
        work['처방금액'] = 0.0

    y = pd.to_numeric(work['처방금액'], errors='coerce').fillna(0.0)
    scores = {}
    for b in ATOMIC_BEHAVIORS:
        x = pd.to_numeric(work[b], errors='coerce').fillna(0.0)
        mean_x = float(np.abs(x).mean())
        if len(work) >= 2 and y.nunique() > 1 and x.nunique() > 1:
            corr = float(np.abs(x.corr(y, method='spearman')))
            if not np.isfinite(corr):
                corr = 0.0
        else:
            corr = 0.0
        scores[b] = max(0.0, corr * (mean_x + 1e-6))

    s = float(sum(scores.values()))
    if s <= 0:
        vols = {b: float(np.abs(pd.to_numeric(work[b], errors='coerce').fillna(0.0)).mean()) for b in ATOMIC_BEHAVIORS}
        v = float(sum(vols.values()))
        if v <= 0:
            return {b: 0.0 for b in ATOMIC_BEHAVIORS}
        return {b: float(vols[b] / v) for b in ATOMIC_BEHAVIORS}
    return {b: float(scores[b] / s) for b in ATOMIC_BEHAVIORS}

def summarize_activity_counts(df_slice, fallback_importance=None):
    """Aggregate 8-behavior activity volumes for detail rendering.
    When source activity is single-label sparse, distribute by fallback importance.
    """
    if df_slice is None or len(df_slice) == 0:
        return {b: 0.0 for b in ATOMIC_BEHAVIORS}
    out = {}
    for b in ATOMIC_BEHAVIORS:
        if b in df_slice.columns:
            out[b] = float(pd.to_numeric(df_slice[b], errors='coerce').fillna(0.0).sum())
        else:
            out[b] = 0.0
    total = float(sum(out.values()))
    if total <= 0:
        return out

    nonzero_behaviors = [b for b, v in out.items() if float(v) > 0]
    if len(nonzero_behaviors) <= 1:
        # Single-label collapse guard: blend observed dominant label with model importance prior
        prior = {}
        for b in ATOMIC_BEHAVIORS:
            v = 0.0
            if isinstance(fallback_importance, dict):
                try:
                    v = float(fallback_importance.get(b, 0.0))
                except Exception:
                    v = 0.0
            if not np.isfinite(v):
                v = 0.0
            prior[b] = max(0.0, v)
        s_prior = float(sum(prior.values()))
        if s_prior <= 0:
            prior = {b: 1.0 / len(ATOMIC_BEHAVIORS) for b in ATOMIC_BEHAVIORS}
        else:
            prior = {b: (prior[b] / s_prior) for b in ATOMIC_BEHAVIORS}

        dominant = nonzero_behaviors[0] if nonzero_behaviors else ATOMIC_BEHAVIORS[0]
        onehot = {b: (1.0 if b == dominant else 0.0) for b in ATOMIC_BEHAVIORS}
        alpha = 0.35  # keep observed activity signal, avoid all-or-nothing zeros
        mixed = {b: (alpha * onehot[b]) + ((1.0 - alpha) * prior[b]) for b in ATOMIC_BEHAVIORS}
        out = {b: float(total * mixed[b]) for b in ATOMIC_BEHAVIORS}
    return out

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
    if '목표금액' in df_raw.columns:
        df_raw['목표금액'] = pd.to_numeric(df_raw['목표금액'], errors='coerce').fillna(0)
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

    def detect_month_col(df):
        for c in ['월', '목표월', 'Month', 'month']:
            if c in df.columns:
                return c
        return None

    def normalize_target_source(df_source, dedupe_hospital=False):
        if df_source.empty or '목표금액' not in df_source.columns:
            return pd.DataFrame(columns=['__k_branch', '__k_rep', '__k_prod', '__month', '지점', '성명', '품목', '목표금액'])
        month_col = detect_month_col(df_source)
        src = df_source.copy()
        if month_col is None:
            src['__month'] = 1
        else:
            src['__month'] = pd.to_numeric(src[month_col], errors='coerce').fillna(1).astype(int)
        for col in ['지점', '성명', '품목']:
            if col not in src.columns:
                src[col] = 'Unknown'
        for k in ['__k_branch', '__k_rep', '__k_prod']:
            if k not in src.columns:
                src[k] = ''
        src['목표금액'] = pd.to_numeric(src['목표금액'], errors='coerce').fillna(0)

        # standardized_sales는 목표값이 거래/일자 단위로 반복될 수 있어 병원 단위 중복을 먼저 축소
        if dedupe_hospital and '병원명' in src.columns:
            src['병원명'] = src['병원명'].astype(str).str.strip()
            src = (
                src.groupby(['__k_branch', '__k_rep', '__k_prod', '__month', '병원명'], as_index=False)
                .agg({
                    '지점': 'first',
                    '성명': 'first',
                    '품목': 'first',
                    '목표금액': 'first'
                })
            )

        src = src[['__k_branch', '__k_rep', '__k_prod', '__month', '지점', '성명', '품목', '목표금액']]
        src = src[src['목표금액'] > 0].copy()
        return src

    has_targets_in_standardized = (
        use_standardized_sales
        and ('목표금액' in df_raw.columns)
        and (pd.to_numeric(df_raw['목표금액'], errors='coerce').fillna(0).sum() > 0)
    )
    target_sources = []
    if has_targets_in_standardized:
        target_sources.append(normalize_target_source(df_raw, dedupe_hospital=True))
    target_sources.append(normalize_target_source(df_targets))

    target_pool = pd.concat(target_sources, ignore_index=True) if target_sources else pd.DataFrame()
    if not target_pool.empty:
        # standardized_sales에 목표가 있을 때 이를 우선 사용하고, 없는 조합만 target 파일로 보강
        if has_targets_in_standardized and len(target_sources) > 1 and not target_sources[1].empty:
            std_keys = set(
                target_sources[0][['__k_branch', '__k_rep', '__k_prod', '__month']]
                .astype(str)
                .agg('|'.join, axis=1)
                .tolist()
            )
            target_from_file = target_sources[1].copy()
            file_keys = (
                target_from_file[['__k_branch', '__k_rep', '__k_prod', '__month']]
                .astype(str).agg('|'.join, axis=1)
            )
            target_from_file = target_from_file[~file_keys.isin(std_keys)]
            target_pool = pd.concat([target_sources[0], target_from_file], ignore_index=True)

        target_pool = (
            target_pool
            .groupby(['__k_branch', '__k_rep', '__k_prod', '__month'], as_index=False)
            .agg({
                '지점': 'first',
                '성명': 'first',
                '품목': 'first',
                '목표금액': 'sum'
            })
        )

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
    if target_pool.empty:
        df_targets_agg = pd.DataFrame(columns=['__k_branch', '__k_rep', '__k_prod', '목표금액'])
    else:
        df_targets_agg = (
            target_pool
            .groupby(['__k_branch', '__k_rep', '__k_prod'], as_index=False)['목표금액']
            .sum()
        )
    df_final = pd.merge(df_master, df_targets_agg, on=['__k_branch','__k_rep','__k_prod'], how='left')
    
    # 누락 데이터 추출 (실적은 있으나 목표가 없는 경우)
    missing_targets_df = df_final[df_final['목표금액'].isna() | (df_final['목표금액'] == 0)]
    missing_log = missing_targets_df[['지점', '성명', '품목']].to_dict('records')
    
    df_final = df_final.fillna(0)
    df_final['달성률'] = np.where(df_final['목표금액'] > 0, (df_final['처방금액'] / df_final['목표금액']) * 100, 0)
    df_final['목표갭'] = df_final['처방금액'] - df_final['목표금액']
    df_final['목표갭률'] = np.where(df_final['목표금액'] > 0, (df_final['처방금액'] / df_final['목표금액'] - 1.0) * 100, 0)
    
    print(f"DEBUG: df_raw shape: {df_raw.shape}")
    print(f"DEBUG: actual_agg shape: {actual_agg.shape}")
    print(f"DEBUG: df_final shape: {df_final.shape}")
    
    if df_final.empty:
        print("[CRITICAL] df_final is empty. There is no matching data between sales and targets.")

    # --- [코칭 룰 엔진] ---
    def get_coaching_message(hir, rtr, bcr, ach, th_hir=70.0, th_rtr=70.0, th_bcr=70.0, th_ach=100.0):
        # 마스터 로직 코칭 룰 (교차 검증 매트릭스)
        if ach >= th_ach:
            if hir >= th_hir and rtr >= th_rtr and bcr >= th_bcr:
                return "The Masterclass", "완벽한 선순환을 만들어내고 있습니다. 현재의 높은 활동량과 우수한 관계 유지 능력을 유지하며 Best Practice 사례로 공유를 권장합니다."
            elif hir >= th_hir and bcr < th_bcr:
                return "The Lucky Hunter", "목표는 달성했으나 몰아치기 영업이 의심됩니다. 방문 규칙성(BCR)을 높여 장기적이고 안정적인 파이프라인 관리가 필요합니다."
            elif hir < th_hir and rtr < th_rtr:
                return "The Data Ghost", "목표를 달성했으나 핵심 활동 데이터(HIR, RTR)가 누락되었거나 요행에 의한 실적일 수 있습니다. 활동 데이터 기록 및 일회성 매출 여부를 점검하세요."
            else:
                return "The Good Performer", "우수한 성과를 달성했습니다. 다만 일부 행동 지표의 개선을 통해 더욱 완벽한 퍼포먼스를 낼 수 있습니다."
        else:
            if hir >= th_hir and rtr >= th_rtr:
                return "The Strategic Sleeper", "현재 우수한 활동량과 관계 지표를 유지하고 있어 곧 실적으로 터질 잠재력이 큽니다. 성과에 조급해하지 말고 현재의 올바른 과정을 꾸준히 지속하세요."
            elif hir < th_hir and rtr < th_rtr:
                return "The Critical Zone", "실적 미달성과 함께 활동량 및 관계 지표가 모두 무너진 심각한 상태입니다. 즉각적인 밀착 코칭 및 파이프라인 전면 재설계가 시급합니다."
            else:
                return "The Hard Worker", "성실하게 활동하고 있으나 성과로 연결되지 않고 있습니다. 효율성 강화를 위해 타겟팅(Segment)이나 주력 품목(MS) 전략의 전면 재점검이 필요합니다."

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
    # target_pool 기반으로 월 타겟을 재구성하여 누락/빈 배열을 방지
    if not target_pool.empty:
        target_monthly = (
            target_pool
            .rename(columns={'__month': '월'})
            .merge(
                df_final[['__k_branch', '__k_rep', '__k_prod', '지점', '성명', '품목']].drop_duplicates(),
                on=['__k_branch', '__k_rep', '__k_prod'],
                how='inner',
                suffixes=('', '_sales')
            )
        )
        for c in ['지점', '성명', '품목']:
            sales_col = f'{c}_sales'
            if sales_col in target_monthly.columns:
                target_monthly[c] = target_monthly[sales_col].fillna(target_monthly[c])
                target_monthly = target_monthly.drop(columns=[sales_col])
    elif target_monthly.empty:
        target_monthly = pd.DataFrame(columns=['__k_branch', '__k_rep', '__k_prod', '월', '지점', '성명', '품목', '목표금액'])

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
    df_rep_raw_calc['REP_ACH'] = np.where(
        df_rep_raw_calc['목표금액'] > 0,
        (df_rep_raw_calc['처방금액'] / df_rep_raw_calc['목표금액']) * 100,
        0
    )
    
    # 절대평가 기준 (T_MEAN 하드코딩)
    th_hir = float(T_MEAN)
    th_rtr = float(T_MEAN)
    th_bcr = float(T_MEAN)
    th_ach = 100.0
    
    print(f"DEBUG: Coaching Thresholds (Absolute) -> HIR:{th_hir:.1f}, RTR:{th_rtr:.1f}, BCR:{th_bcr:.1f}, ACH:{th_ach:.1f}")

    for br in df_final['지점'].unique():
        df_br = df_final[df_final['지점'] == br]
        df_br_raw = df_raw[df_raw['지점'] == br]
        hierarchy['branches'][br] = {
            'members': [],
            'avg': df_br[['HIR', 'RTR', 'BCR', 'PHR']].mean().to_dict(),
            'achieve': calc_achieve(df_br['처방금액'].sum(), df_br['목표금액'].sum()),
            'actual_sum': float(df_br['처방금액'].sum()),
            'target_sum': float(df_br['목표금액'].sum()),
            'gap_amount': float(calc_gap(df_br['처방금액'].sum(), df_br['목표금액'].sum())[0]),
            'gap_rate': float(calc_gap(df_br['처방금액'].sum(), df_br['목표금액'].sum())[1]),
            'monthly_actual': df_raw[df_raw['지점'] == br].groupby('월')['처방금액'].sum().reindex(month_axis, fill_value=0).tolist(),
            'monthly_target': target_monthly[target_monthly['지점'] == br].groupby('월')['목표금액'].sum().reindex(month_axis, fill_value=0).tolist(),
            'analysis': run_full_analysis(df_br_raw),
            'prod_analysis': {pd: {
                'analysis': run_full_analysis(df_br_raw[df_br_raw['품목']==pd]),
                'monthly_actual': df_raw[(df_raw['지점'] == br) & (df_raw['품목'] == pd)].groupby('월')['처방금액'].sum().reindex(month_axis, fill_value=0).tolist(),
                'monthly_target': target_monthly[(target_monthly['지점'] == br) & (target_monthly['품목'] == pd)].groupby('월')['목표금액'].sum().reindex(month_axis, fill_value=0).tolist(),
                'achieve': calc_achieve(df_br[df_br['품목']==pd]['처방금액'].sum(), df_br[df_br['품목']==pd]['목표금액'].sum()),
                'actual_sum': float(df_br[df_br['품목']==pd]['처방금액'].sum()),
                'target_sum': float(df_br[df_br['품목']==pd]['목표금액'].sum()),
                'gap_amount': float(calc_gap(df_br[df_br['품목']==pd]['처방금액'].sum(), df_br[df_br['품목']==pd]['목표금액'].sum())[0]),
                'gap_rate': float(calc_gap(df_br[df_br['품목']==pd]['처방금액'].sum(), df_br[df_br['품목']==pd]['목표금액'].sum())[1]),
                'avg': df_br[df_br['품목']==pd][['HIR','RTR','BCR','PHR']].mean().to_dict()
            } for pd in hierarchy['products']}
        }
        
        for rep in df_br['성명'].unique():
            df_rep = df_br[df_br['성명'] == rep]
            rep_raw = df_raw[(df_raw['지점'] == br) & (df_raw['성명'] == rep)]
            rep_analysis = run_full_analysis(rep_raw)
            if rep_analysis is not None:
                real_shap = {k: float(v) for k, v in rep_analysis['importance'].items()}
            else:
                real_shap = {b: np.nan for b in ATOMIC_BEHAVIORS}
            
            prod_matrix = []
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

            rep_prod_analysis = {}
            rep_total_sales = float(pd.to_numeric(rep_raw['처방금액'], errors='coerce').fillna(0.0).sum()) if '처방금액' in rep_raw.columns else 0.0
            rep_total_act = {b: float(pd.to_numeric(rep_raw[b], errors='coerce').fillna(0.0).sum()) if b in rep_raw.columns else 0.0 for b in ATOMIC_BEHAVIORS}
            rep_activity_counts = summarize_activity_counts(rep_raw, real_shap)
            for pd_name in hierarchy['products']:
                df_rep_prod = df_rep[df_rep['품목'] == pd_name]
                df_rep_raw_prod = rep_raw[rep_raw['품목'] == pd_name]
                rep_prod_ana = run_full_analysis(df_rep_raw_prod)
                if rep_prod_ana is not None and (rep_prod_ana or {}).get('importance'):
                    rep_prod_shap = {k: float(v) for k, v in rep_prod_ana.get('importance', {}).items()}
                else:
                    weighted = {}
                    prod_sales = float(pd.to_numeric(df_rep_prod['처방금액'], errors='coerce').fillna(0.0).sum()) if '처방금액' in df_rep_prod.columns else 0.0
                    sales_share = (prod_sales / rep_total_sales) if rep_total_sales > 0 else 0.0
                    for b in ATOMIC_BEHAVIORS:
                        base_imp = float((real_shap or {}).get(b, 0.0))
                        prod_act = float(pd.to_numeric(df_rep_raw_prod[b], errors='coerce').fillna(0.0).sum()) if b in df_rep_raw_prod.columns else 0.0
                        total_act = float(rep_total_act.get(b, 0.0))
                        act_share = (prod_act / total_act) if total_act > 0 else 0.0
                        mix_share = 0.5 * act_share + 0.5 * sales_share
                        weighted[b] = max(0.0, base_imp * mix_share)
                    if sum(weighted.values()) > 0:
                        rep_prod_shap = {b: float(weighted[b]) for b in ATOMIC_BEHAVIORS}
                    else:
                        rep_prod_shap = estimate_atomic_importance(df_rep_raw_prod)
                if sum(abs(float(v or 0.0)) for v in rep_prod_shap.values()) <= 0:
                    rep_prod_shap = {k: float(v) for k, v in (real_shap or {}).items()}
                rep_prod_analysis[pd_name] = {
                    'analysis': rep_prod_ana,
                    'shap': rep_prod_shap,
                    'activity_counts': summarize_activity_counts(df_rep_raw_prod, rep_prod_shap),
                    'achieve': calc_achieve(df_rep_prod['처방금액'].sum(), df_rep_prod['목표금액'].sum()),
                    'actual_sum': float(df_rep_prod['처방금액'].sum()),
                    'target_sum': float(df_rep_prod['목표금액'].sum()),
                    'gap_amount': float(calc_gap(df_rep_prod['처방금액'].sum(), df_rep_prod['목표금액'].sum())[0]),
                    'gap_rate': float(calc_gap(df_rep_prod['처방금액'].sum(), df_rep_prod['목표금액'].sum())[1]),
                    'avg': df_rep_prod[['HIR', 'RTR', 'BCR', 'PHR']].mean().to_dict(),
                    'HIR': float(df_rep_prod['HIR'].mean()) if not df_rep_prod.empty else 0.0,
                    'RTR': float(df_rep_prod['RTR'].mean()) if not df_rep_prod.empty else 0.0,
                    'BCR': float(df_rep_prod['BCR'].mean()) if not df_rep_prod.empty else 0.0,
                    'PHR': float(df_rep_prod['PHR'].mean()) if not df_rep_prod.empty else 0.0,
                    'monthly_actual': df_raw[(df_raw['지점'] == br) & (df_raw['성명'] == rep) & (df_raw['품목'] == pd_name)]
                        .groupby('월')['처방금액'].sum().reindex(month_axis, fill_value=0).tolist(),
                    'monthly_target': target_monthly[(target_monthly['지점'] == br) & (target_monthly['성명'] == rep) & (target_monthly['품목'] == pd_name)]
                        .groupby('월')['목표금액'].sum().reindex(month_axis, fill_value=0).tolist(),
                }

            hierarchy['branches'][br]['members'].append({
                '성명': rep,
                'HIR': rep_hir, 'RTR': rep_rtr, 'BCR': rep_bcr, 'PHR': float(df_rep['PHR'].mean()),
                '처방금액': float(df_rep['처방금액'].sum()), '목표금액': float(df_rep['목표금액'].sum()),
                'achieve': calc_achieve(df_rep['처방금액'].sum(), df_rep['목표금액'].sum()),
                'gap_amount': float(calc_gap(df_rep['처방금액'].sum(), df_rep['목표금액'].sum())[0]),
                'gap_rate': float(calc_gap(df_rep['처방금액'].sum(), df_rep['목표금액'].sum())[1]),
                '지점순위': int(df_br.groupby('성명')['처방금액'].sum().rank(ascending=False)[rep]),
                'shap': real_shap,
                'coach_scenario': c_name,
                'coach_action': c_action,
                'efficiency': float(df_rep['처방금액'].sum() / (rep_raw['HIR_W'].sum() + 1)) if not rep_raw.empty else 0.0,
                'gini': float(calc_gini(df_rep['처방금액'])),
                'avg_ms': avg_ms,
                'prod_matrix': prod_matrix,
                'activity_counts': rep_activity_counts,
                'prod_analysis': rep_prod_analysis,
                'monthly_actual': df_raw[(df_raw['지점']==br) & (df_raw['성명']==rep)].groupby('월')['처방금액'].sum().reindex(month_axis, fill_value=0).tolist(),
                'monthly_target': target_monthly[(target_monthly['지점']==br) & (target_monthly['성명']==rep)].groupby('월')['목표금액'].sum().reindex(month_axis, fill_value=0).tolist()
            })

    hierarchy['total_prod_analysis'] = { pd: {
        'analysis': run_full_analysis(df_raw[df_raw['품목']==pd]),
        'monthly_actual': df_raw[df_raw['품목']==pd].groupby('월')['처방금액'].sum().reindex(month_axis, fill_value=0).tolist(),
        'monthly_target': target_monthly[target_monthly['품목']==pd].groupby('월')['목표금액'].sum().reindex(month_axis, fill_value=0).tolist(),
        'achieve': calc_achieve(df_final[df_final['품목']==pd]['처방금액'].sum(), df_final[df_final['품목']==pd]['목표금액'].sum()),
        'actual_sum': float(df_final[df_final['품목']==pd]['처방금액'].sum()),
        'target_sum': float(df_final[df_final['품목']==pd]['목표금액'].sum()),
        'gap_amount': float(calc_gap(df_final[df_final['품목']==pd]['처방금액'].sum(), df_final[df_final['품목']==pd]['목표금액'].sum())[0]),
        'gap_rate': float(calc_gap(df_final[df_final['품목']==pd]['처방금액'].sum(), df_final[df_final['품목']==pd]['목표금액'].sum())[1]),
        'avg': df_final[df_final['품목']==pd][['HIR','RTR','BCR','PHR']].mean().to_dict()
    } for pd in hierarchy['products']}

    hierarchy['total'] = {
        'analysis': run_full_analysis(df_raw), 'avg': hierarchy['total_avg'],
        'monthly_actual': df_raw.groupby('월')['처방금액'].sum().reindex(month_axis, fill_value=0).tolist(),
        'monthly_target': target_monthly.groupby('월')['목표금액'].sum().reindex(month_axis, fill_value=0).tolist(),
        'achieve': calc_achieve(df_final['처방금액'].sum(), df_final['목표금액'].sum()),
        'actual_sum': float(df_final['처방금액'].sum()),
        'target_sum': float(df_final['목표금액'].sum()),
        'gap_amount': float(calc_gap(df_final['처방금액'].sum(), df_final['목표금액'].sum())[0]),
        'gap_rate': float(calc_gap(df_final['처방금액'].sum(), df_final['목표금액'].sum())[1])
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
    pattern = r'const\s+db\s*=\s*/\*DATA_JSON_PLACEHOLDER\*/\s*.*?;'
    replacement = f'const db = {total_json};'
    
    if re.search(pattern, template, flags=re.S):
        template = re.sub(pattern, replacement, template, count=1, flags=re.S)
        print("[INFO] 템플릿 데이터 주입 완료 (정규표현식 매칭)")
    elif '/*DATA_JSON_PLACEHOLDER*/' in template:
        # 정규표현식이 실패할 경우를 대비한 단순 문자열 교체 시도
        # 템플릿의 초기 객체 구조와 상관없이 주석 위치를 기준으로 교체
        template = re.sub(r'/\*DATA_JSON_PLACEHOLDER\*/\s*.*?;', f'{total_json};', template, count=1, flags=re.S)
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
