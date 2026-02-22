import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os
import glob
import sys

# scripts 폴더를 경로에 추가하여 모듈 임포트 가능하게 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- [1. 기본 환경 설정] ---
st.set_page_config(layout="wide", page_title="SFE Master Sandbox V13.1")

# --- [유니크 파일명 생성 유틸리티] ---
def get_unique_filename(base_dir, base_name, ext):
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

# --- [유틸리티: 필드 매핑 로드/저장] ---
def load_mapping_config():
    import json
    config_path = 'config/mapping.json'
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "지점": ["지점", "지점명", "Branch"], 
        "성명": ["성명", "담당자명", "Rep", "담당자"], 
        "병원명": ["병원명", "거래처명", "요양기관명", "Hospital", "거래처"],
        "품목": ["품목", "품목명", "Product"],
        "처방금액": ["처방금액", "실적금액", "Amount", "실적"], 
        "목표금액": ["목표금액", "Target"],
        "월": ["월", "목표월", "Month"], 
        "activities": ["activities", "활동", "활동명"], 
        "segment": ["segment", "규모", "종별코드명"],
        "날짜": ["날짜", "활동일자", "목표월", "Date"]
    }

def save_mapping_config(mapping_dict):
    import json
    config_path = 'config/mapping.json'
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, ensure_ascii=False, indent=2)

def find_best_match(target_key, available_cols, mapping_dict):
    aliases = mapping_dict.get(target_key, [])
    for i, col in enumerate(available_cols):
        if col in aliases or col.lower() in [a.lower() for a in aliases]:
            return i
    return 0

# --- [2. 6대 마스터 로직 엔진 (Detailed Analytics)] ---
def calculate_master_engine(df, cfg):
    """
    구재현 님의 12대 마스터 로직을 데이터프레임에 주입합니다.
    """
    # [Logic 1] HIR & PI 가중치 기초 매핑
    df['W_Act'] = df['activities'].map(cfg['hir_weights']).fillna(1.0)
    df['W_Seg'] = df['segment'].map(cfg['pi_weights']).fillna(1.0)
    
    # [지표 1] HIR (High-Impact Rate) - 활동의 질적 평가
    # 수식: (가중치 * 품질) / 총 활동
    df['HIR_Raw'] = df['W_Act'] * 1.0 # 품질 점수는 1.0 기본값 처리
    
    # [지표 2] RTR (Relationship Temp) - 시간 감쇠 로직
    # 수식: Sentiment * exp(-lambda * t)
    max_date = df['날짜'].max()
    df['days_diff'] = (max_date - df['날짜']).dt.days
    df['RTR_Raw'] = np.exp(-cfg['rtr_lambda'] * df['days_diff'])
    
    # [지표 3] BCR (Behavior Consistency) - 활동 규칙성
    # 샌드박스에서는 행별 빈도를 카운트하여 집계 시점에 표준편차 연산 준비
    df['BCR_Raw'] = 1.0 
    
    # [지표 4] PHR (Pipeline Health) - 전략 활동 여부
    df['PHR_Raw'] = df['activities'].apply(lambda x: 1.0 if x in cfg['phr_acts'] else 0.0)
    
    # [지표 5] FGR (Field Growth Rate) - Q와 P의 밸런스
    # 집계 시점에서 Q(60%) + P(40%) 가중치 적용 예정
    
    # [지표 6] PI (Prescription Index) - 난이도 보정 성과지수
    # 수식: 가중Rx * 0.7 + 성장률 * 0.3
    df['PI_Raw'] = df['처방수량'] * df['W_Seg']
    
    return df

# --- [3. 사이드바: 6대 지표 정밀 전략 설정] ---
with st.sidebar:
    st.header("⚙️ 6대 지표 마스터 로직 설정")
    
    # 1. HIR 설정
    with st.expander("1. HIR (활동 가중치)", expanded=True):
        w_pt = st.slider("PT(설명회)", 1.0, 5.0, 3.5, 0.1)
        w_demo = st.slider("시연(Demo)", 1.0, 5.0, 3.0, 0.1)
        w_close = st.slider("클로징(Closing)", 1.0, 5.0, 4.0, 0.1)
        w_visit = st.slider("일반대면(Visit)", 1.0, 5.0, 2.0, 0.1)
        HIR_W = {'PT': w_pt, '시연': w_demo, '클로징': w_close, '대면': w_visit, 
                 '니즈환기': 1.5, '컨택': 1.2, '접근': 1.0, '피드백': 1.0}

    # 2. RTR 설정
    with st.expander("2. RTR (관계 온도 감쇠)"):
        r_lam = st.number_input("감쇠상수(λ)", 0.001, 0.100, 0.035, format="%.3f", help="값이 클수록 관계가 빨리 식음")
        
    # 3. PHR 설정
    with st.expander("3. PHR (파이프라인 기준)"):
        phr_list = st.multiselect("전략 활동(Next Action) 정의", 
                                 options=['PT', '시연', '클로징', '니즈환기', '대면'], 
                                 default=['PT', '시연', '클로징'])

    # 4. FGR 설정
    with st.expander("4. FGR (시장지배력 가중치)"):
        fgr_q_ratio = st.slider("처방수량(Q) 반영비중", 0.0, 1.0, 0.6)
        
    # 5. PI 설정
    with st.expander("5. PI (병원 난이도 보정)"):
        w_tertiary = st.number_input("상급종합 가중치", 1.0, 2.0, 1.5)
        w_general = st.number_input("종합병원 가중치", 1.0, 2.0, 1.2)
        PI_W = {'상급종합': w_tertiary, '종합병원': w_general, '일반의원': 1.0, '약국/기타': 0.8}

    CONFIG = {
        'hir_weights': HIR_W, 'rtr_lambda': r_lam, 
        'phr_acts': phr_list, 'fgr_q_w': fgr_q_ratio, 'pi_weights': PI_W
    }

# --- [4. 메인 화면: 다중 파일 통합 및 매핑] ---
st.title("🧪 SFE Agile Sandbox V13.1")
st.markdown("##### [정제] 다중 파일 통합 및 표준화 ➔ [전략] 마스터 로직 주입 ➔ [배포] 빌더용 CSV 추출")

if 'clean_master' not in st.session_state:
    st.session_state.clean_master = None

with st.expander("📂 STEP 1. 데이터 선택 및 통합", expanded=True):
    # 자동으로 데이터 폴더 스캔
    data_dirs = ['data/sales', 'data/targets', 'data/crm']
    available_files = []
    for d in data_dirs:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, "*.csv")) + glob.glob(os.path.join(d, "*.xlsx"))
            available_files.extend(files)
    
    st.info(f"🔍 시스템이 {len(available_files)}개의 분석 가능한 파일을 찾았습니다.")
    
    selected_files = st.multiselect(
        "분석에 포함할 파일을 선택하세요", 
        options=available_files,
        default=available_files[:1] if available_files else [],
        help="data 폴더 내의 파일들이 자동으로 표시됩니다."
    )
    
    # 추가 업로드 기능 유지
    uploaded_files = st.file_uploader("그 외 추가로 업로드할 파일이 있다면 선택하세요", type=["csv", "xlsx"], accept_multiple_files=True)
    
    all_data_sources = selected_files + (uploaded_files if uploaded_files else [])
    
    if all_data_sources:
        # 파일 통합 로직
        df_list = []
        for f in all_data_sources:
            if isinstance(f, str): # 경로 문자열인 경우 (자동 탐색)
                if f.endswith('.xlsx'):
                    df_list.append(pd.read_excel(f))
                else:
                    df_list.append(pd.read_csv(f))
            else: # 업로드된 파일 객체인 경우
                if f.name.endswith('.xlsx'):
                    df_list.append(pd.read_excel(f))
                else:
                    df_list.append(pd.read_csv(f))
        
        raw_df = pd.concat(df_list, ignore_index=True)
        st.success(f"✅ 총 {len(all_data_sources)}개 데이터 소스 통합 완료 (총 {len(raw_df):,}건)")
        
        # 매핑 폼
        cols = raw_df.columns.tolist()
        mapping_config = load_mapping_config()
        
        with st.form("master_mapping"):
            st.info("💡 시스템이 등록된 별명 사전을 기반으로 컬럼을 자동 매핑했습니다. 올바르지 않다면 직접 선택하세요.")
            c1, c2, c3 = st.columns(3)
            with c1:
                m_br = st.selectbox("지점(Branch)", options=cols, index=find_best_match("지점", cols, mapping_config))
                m_rep = st.selectbox("담당자(Rep)", options=cols, index=find_best_match("성명", cols, mapping_config))
            with c2:
                m_hosp = st.selectbox("병원명(Hospital)", options=cols, index=find_best_match("병원명", cols, mapping_config))
                m_pd = st.selectbox("품목(Product)", options=cols, index=find_best_match("품목", cols, mapping_config))
                m_val = st.selectbox("실적(Amount)", options=cols, index=find_best_match("처방금액", cols, mapping_config))
            with c3:
                m_act = st.selectbox("활동(Activity)", options=cols, index=find_best_match("activities", cols, mapping_config))
                m_dt = st.selectbox("날짜(Date)", options=cols, index=find_best_match("날짜", cols, mapping_config))
                m_seg = st.selectbox("세그먼트(Segment)", options=cols, index=find_best_match("segment", cols, mapping_config))
            
            learn_mapping = st.checkbox("이 매핑 정보를 별명 사전에 추가하여 학습하기", value=True)
            
            if st.form_submit_button("🚀 마스터 로직 적용 및 데이터 표준화"):
                # 학습 모드: 새로운 별명이면 저장
                if learn_mapping:
                    updated = False
                    mapping_pairs = [("지점", m_br), ("성명", m_rep), ("병원명", m_hosp), ("품목", m_pd), ("처방금액", m_val), ("activities", m_act), ("날짜", m_dt), ("segment", m_seg)]
                    for key, val in mapping_pairs:
                        if key in mapping_config and val not in mapping_config[key]:
                            mapping_config[key].append(val)
                            updated = True
                    if updated:
                        save_mapping_config(mapping_config)
                        st.toast("🧠 시스템이 새로운 컬럼 매핑을 학습했습니다!", icon="⚡")

                # 컬럼명 표준화 (안전한 매핑)
                rename_map = {
                    m_br: '지점', m_rep: '성명', m_hosp: '병원명', m_pd: '품목',
                    m_val: '처방금액', m_act: 'activities', 
                    m_dt: '날짜', m_seg: 'segment'
                }
                
                # 중복된 매핑 제거 (동일한 원본 컬럼이 다른 이름으로 두 번 매핑될 때 마지막 것 유지)
                df_std = raw_df.copy()
                df_std = df_std.rename(columns=rename_map)

                # 사용자 선택 컬럼을 최우선으로 activities에 강제 반영
                if m_act in raw_df.columns:
                    df_std['activities'] = raw_df[m_act]

                # 활동명 매핑 보정:
                # rename 이후 컬럼이 바뀌어도 raw_df의 원본 활동 컬럼을 우선 사용해 activities를 복구한다.
                activity_source_cols = []
                if m_act in raw_df.columns:
                    activity_source_cols.append(m_act)
                for col in raw_df.columns:
                    col_str = str(col)
                    col_esc = col_str.encode('unicode_escape').decode()
                    if col_str in ['activities', 'activity', 'Activity'] or ('\\ud65c\\ub3d9' in col_esc):
                        activity_source_cols.append(col)
                activity_source_cols = list(dict.fromkeys(activity_source_cols))

                if activity_source_cols:
                    act_series = pd.Series(np.nan, index=raw_df.index, dtype='object')
                    for col in activity_source_cols:
                        src = raw_df[col]
                        src = src.where(src.notna() & (src.astype(str).str.strip() != ''), np.nan)
                        act_series = act_series.fillna(src)
                    df_std['activities'] = act_series

                # 선택한 activity 컬럼을 "병합" 형태로 보강:
                # CRM 행에만 있는 활동값을 키 기준으로 전체 행에 전파한다.
                if 'activities' in df_std.columns:
                    df_std['activities'] = df_std['activities'].astype('object')
                    df_std['activities'] = df_std['activities'].where(
                        df_std['activities'].notna() & (df_std['activities'].astype(str).str.strip() != ''),
                        np.nan
                    )

                    # 월 키 생성 (날짜/활동일자/목표월/월 중 가용 컬럼 사용)
                    month_src = None
                    for c in ['날짜', '활동일자', '목표월', '월']:
                        if c in df_std.columns:
                            month_src = c
                            break
                    if month_src is not None:
                        parsed_month = pd.to_datetime(df_std[month_src], errors='coerce').dt.month
                        if parsed_month.notna().sum() <= len(df_std) * 0.3:
                            parsed_month = pd.to_numeric(df_std[month_src], errors='coerce')
                        df_std['__act_month'] = parsed_month
                    else:
                        df_std['__act_month'] = np.nan

                    key_candidates = ['지점', '성명', '품목', '병원ID', '__act_month']
                    merge_keys = [k for k in key_candidates if k in df_std.columns]
                    merge_keys = [k for k in merge_keys if not df_std[k].isna().all()]

                    if merge_keys:
                        donors = df_std[df_std['activities'].notna()].copy()
                        if not donors.empty:
                            valid_keys = [k for k in merge_keys if donors[k].notna().sum() > 0 and len(donors[donors[k] != 'nan']) > 0]
                            if valid_keys:
                                for k in valid_keys:
                                    if donors[k].dtype == object:
                                        donors[k] = donors[k].astype(str).str.strip().replace('nan', np.nan)
                                        df_std[k] = df_std[k].astype(str).str.strip().replace('nan', np.nan)
                                
                                act_map = (
                                    donors.groupby(valid_keys)['activities']
                                    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
                                    .reset_index()
                                    .rename(columns={'activities': '__activities_mapped'})
                                )
                                df_std = df_std.merge(act_map, on=valid_keys, how='left')
                                df_std['activities'] = df_std['activities'].fillna(df_std['__activities_mapped'])
                                df_std = df_std.drop(columns=['__activities_mapped'])

                    if '__act_month' in df_std.columns:
                        df_std = df_std.drop(columns=['__act_month'])

                if 'activities' in df_std.columns and df_std['activities'].notna().sum() == 0:
                    st.warning("활동(Activity) 매핑 결과가 비어 있습니다. CRM의 '활동명' 컬럼 선택 여부를 확인하세요.")
                
                # 디버깅: 매핑 결과 확인
                if '날짜' not in df_std.columns:
                    st.error(f"❌ '날짜' 컬럼을 찾을 수 없습니다. (선택된 원본: {m_dt})")
                    st.write("현재 컬럼 리스트:", df_std.columns.tolist())
                    st.stop()

                df_std['날짜'] = pd.to_datetime(df_std['날짜'], errors='coerce')
                # 결측치 제거
                df_std = df_std.dropna(subset=['날짜'])
                
                # 처방수량 부재 시 처방금액 기반 가상 생성 (분석용)
                if '처방수량' not in df_std.columns:
                    amt = pd.to_numeric(df_std['처방금액'], errors='coerce')
                    qty = (amt / 1000).replace([np.inf, -np.inf], np.nan).fillna(0)
                    df_std['처방수량'] = qty.astype(int)
                
                # 💡 6대 지표 마스터 엔진 가동
                st.session_state.clean_master = calculate_master_engine(df_std, CONFIG)
                
                # [자동 저장] 빌더 연동을 위해 물리적 파일로 즉시 저장
                output_dir = 'output/processed_data'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # 가공 데이터 파일명 규칙 적용
                save_path = get_unique_filename(output_dir, 'standardized_sales', 'csv')
                st.session_state.clean_master.to_csv(save_path, index=False, encoding='utf-8-sig')
                st.success(f"✨ 모든 분석 로직이 주입되었으며, '{save_path}'로 자동 저장되었습니다.")

# --- [5. 데이터 검증 및 리포트 빌더 연동] ---
if st.session_state.clean_master is not None:
    df = st.session_state.clean_master
    st.divider()
    
    st.subheader("📊 STEP 2. 전략 데이터 검증 및 추출")
    
    # 탭 생성
    tab1, tab2 = st.tabs(["📊 1차 결과 템플릿", "🗺️ 전국병원 지도 뷰"])
    
    with tab1:
        # 지표 요약 시각화 (Ad-hoc)
        t1, t2 = st.columns([1, 3])
        with t1:
            st.write("🔍 즉석 데이터 확인")
            view_dim = st.selectbox("분석 차원", ['지점', '성명', '품목'])
            view_metric = st.selectbox("분석 지표", ['처방금액', 'HIR_Raw', 'RTR_Raw', 'PHR_Raw'])
        with t2:
            view_df = df.groupby(view_dim)[view_metric].mean().reset_index()
            fig = px.bar(view_df, x=view_dim, y=view_metric, template='plotly_white', color=view_metric)
            st.plotly_chart(fig, use_container_width=True)
    
        # 📦 리포트 빌더용 파일 추출 섹션
        st.info("📦 **리포트 빌더 및 최종 결과물 생성**")
        final_cols = ['지점', '성명', '병원명', '품목', '처방금액', '처방수량', 'activities', 'segment', '날짜', 'HIR_Raw', 'RTR_Raw', 'PHR_Raw', 'PI_Raw']
        export_df = df[[c for c in final_cols if c in df.columns]]
        
        c1, c2 = st.columns(2)
        with c1:
            csv_out = export_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 표준 CSV 다운로드",
                data=csv_out,
                file_name="standardized_sales.csv",
                mime="text/csv",
                help="이 파일을 다운로드하여 별도로 보관할 수 있습니다."
            )
        
        with c2:
            if st.button("🛠️ 최종 전략 리포트(HTML) 생성", type="primary"):
                with st.spinner("🚀 고차원 분석 엔진 가동 중..."):
                    try:
                        # report_builder_v12의 로직 호출 (현재 슬라이더 설정 반영)
                        from report_builder_v12 import build_final_reports
                        output_file = build_final_reports(external_config=CONFIG)
                        
                        if output_file:
                            st.success(f"✅ 리포트 생성 완료! \n\n 파일 위치: `{output_file}`")
                            
                            # 생성된 HTML 파일을 바로 다운로드할 수 있게 제공
                            with open(output_file, "rb") as f:
                                st.download_button(
                                    label="🚀 생성된 대시보드 바로 다운로드",
                                    data=f,
                                    file_name=os.path.basename(output_file),
                                    mime="text/html"
                                )
                    except Exception as e:
                        st.error(f"❌ 리포트 생성 중 오류 발생: {str(e)}")
        
        st.divider()
        st.dataframe(export_df.head(20))

    with tab2:
        st.markdown("#### 🗺️ 전국병원 지도 뷰")
        st.info("기존 병원 마커 위에 현재 실적 데이터가 오버레이 됩니다. (파이썬에서 지도를 매번 연산하지 않고, 프론트엔드 단에서 DOM을 조작하여 데이터를 덮어씌웁니다.)")
        
        map_path = os.path.join("c:\\", "agent_b", "hospital_map.html")
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                html_data = f.read()
                
            # --- [데이터 오버레이 로직] ---
            if '병원명' in df.columns:
                hosp_df = df.groupby('병원명').agg({
                    '처방금액': 'sum',
                    '처방수량': 'sum',
                    '성명': lambda x: ', '.join(x.dropna().unique())
                }).reset_index()
                
                import json
                overlay_data = {}
                for _, row in hosp_df.iterrows():
                    h_key = str(row['병원명']).strip()
                    if h_key and h_key != 'nan':
                        overlay_data[h_key] = {
                            "처방금액": int(row['처방금액']) if pd.notnull(row['처방금액']) else 0,
                            "처방수량": int(row['처방수량']) if pd.notnull(row['처방수량']) else 0,
                            "담당자": str(row['성명']).strip()
                        }
                
                overlay_json = json.dumps(overlay_data, ensure_ascii=False)
                
                # HTML 템플릿에 스크립트 삽입
                inject_script = f"""
                <script>
                (function() {{
                    const overlayData = {overlay_json};
                    const overlayKeys = Object.keys(overlayData);
                    let matchCount = 0;
                    let markersFound = 0;
                    let attempts = 0;

                    function startOverlay() {{
                        attempts++;
                        let allMarkers = [];
                        
                        // 1. 모든 전역 객체 뒤져서 마커/클러스터/맵 찾기
                        for (let key in window) {{
                            let obj = window[key];
                            if (!obj) continue;
                            
                            // 맵 또는 클러스터 그룹인 경우
                            if (obj.eachLayer && (key.startsWith('map_') || key.startsWith('marker_cluster_'))) {{
                                obj.eachLayer(layer => {{
                                    if (layer.getTooltip) allMarkers.push(layer);
                                    if (layer.eachLayer) {{ // 클러스터 내부 재탐색
                                        try {{
                                            layer.eachLayer(sub => {{ if(sub.getTooltip) allMarkers.push(sub); }});
                                        }} catch(e) {{}}
                                    }}
                                }});
                            }}
                            
                            // 개별 마커인 경우 (marker_...)
                            if (key.startsWith('marker_') && obj.getTooltip) {{
                                allMarkers.push(obj);
                            }}
                        }}
                        
                        // 중복 제거
                        allMarkers = [...new Set(allMarkers)];
                        markersFound = allMarkers.length;

                        if (markersFound === 0 && attempts < 10) {{
                            // 아직 지도가 안 그려졌으면 1초 뒤 재시도
                            setTimeout(startOverlay, 1000);
                            updateStatus("⏳ 지도를 로드 중입니다... (" + attempts + "/10)");
                            return;
                        }}

                        allMarkers.forEach(marker => {{
                            const tt = marker.getTooltip();
                            if (!tt) return;
                            
                            const content = tt.getContent();
                            const div = document.createElement('div');
                            div.innerHTML = content;
                            const hospName = div.innerText.trim();
                            const hospNorm = hospName.replace(/\\s+/g, '').toLowerCase();
                            
                            let matchedKey = null;
                            for(let i=0; i<overlayKeys.length; i++) {{
                                let keyNorm = overlayKeys[i].replace(/\\s+/g, '').toLowerCase();
                                if(hospNorm.indexOf(keyNorm) !== -1 || keyNorm.indexOf(hospNorm) !== -1) {{
                                    matchedKey = overlayKeys[i];
                                    break;
                                }}
                            }}
                            
                            if (matchedKey) {{
                                matchCount++;
                                const d = overlayData[matchedKey];
                                
                                // 🌟 Streamlit 연동: 마커 클릭 시 병원명 전송
                                marker.on('click', function() {{
                                    if (typeof window.setComponentValue === 'function') {{
                                        window.setComponentValue(matchedKey);
                                    }}
                                }});

                                const pop = marker.getPopup();
                                if (pop) {{
                                    const appendHtml = "<hr><h5 style='color:#0d6efd; font-weight:bold; margin-top:10px;'>🔹 실적 달성 현황</h5>" +
                                                     "<div style='font-size:13px;'>" +
                                                     "<b>실적금액:</b> <span style='color:red;'>" + d.처방금액.toLocaleString() + " 원</span><br>" +
                                                     "<b>담당자:</b> " + d.담당자 + "</div>";
                                    
                                    const currentContent = pop.getContent();
                                    if (typeof currentContent === 'string' && currentContent.indexOf('실적 달성 현황') === -1) {{
                                        pop.setContent(currentContent.replace('</div>', appendHtml + '</div>'));
                                    }} else if (currentContent instanceof HTMLElement && currentContent.innerHTML.indexOf('실적 달성 현황') === -1) {{
                                        currentContent.innerHTML += appendHtml;
                                    }}
                                    
                                    if (window.L && L.AwesomeMarkers) {{
                                        marker.setIcon(L.AwesomeMarkers.icon({{
                                            markerColor: 'green', iconColor: 'white', icon: 'star', prefix: 'fa'
                                        }}));
                                    }}
                                }}
                            }}
                        }});

                        updateStatus("<b>✅ 오버레이 완료</b><br>찾은 마커: " + markersFound + "<br>매칭 성공: " + matchCount);
                    }}

                    function updateStatus(msg) {{
                        let d = document.getElementById('debug-box');
                        if (!d) {{
                            d = document.createElement('div');
                            d.id = 'debug-box';
                            d.style.cssText = "position:absolute;top:10px;left:50px;z-index:9999;background:white;padding:12px;border:2px solid #0d6efd;border-radius:10px;font-family:sans-serif;box-shadow:0 4px 10px rgba(0,0,0,0.2);min-width:150px;";
                            document.body.appendChild(d);
                        }}
                        d.innerHTML = "<b>🔍 분석 엔진 가동</b><br>" + 
                                     "<span style='font-size:12px;'>데이터 병원: " + overlayKeys.length + "개</span><br>" + msg;
                    }}

                    // 초기 실행
                    setTimeout(startOverlay, 2000);
                    
                    // --- Streamlit Component Bridge ---
                    function sendMsg(type, data) {{
                        var outData = Object.assign({{isStreamlitMessage: true, type: type}}, data);
                        window.parent.postMessage(outData, "*");
                    }}
                    sendMsg("streamlit:componentReady", {{apiVersion: 1}});
                    sendMsg("streamlit:setFrameHeight", {{height: 750}});
                    window.setComponentValue = function(val) {{
                        sendMsg("streamlit:setComponentValue", {{value: val}});
                    }};
                }})();
                </script>
                """
                html_data += inject_script
            else:
                st.warning("⚠️ '병원명' 매핑이 안 되었습니다. STEP 1에서 요양기관명을 '병원명'으로 선택해주세요.")
            # ------------------------------
            
            # --- [Component 렌더링 및 딥다이브 연동] ---
            map_dir = os.path.join(current_dir, "map_component")
            os.makedirs(map_dir, exist_ok=True)
            with open(os.path.join(map_dir, "index.html"), "w", encoding="utf-8") as f:
                f.write(html_data)
                
            from streamlit.components.v1 import declare_component
            map_component = declare_component("hospital_map", path=map_dir)
            
            c1, c2 = st.columns([7, 3])
            
            with c1:
                st.markdown("##### 📍 전국 병원 분포 (마커를 클릭하세요)")
                clicked_hosp = map_component(key="hosp_map_comp")
                
            with c2:
                st.markdown("### 🔍 딥다이브 플래시보드")
                if clicked_hosp:
                    st.success(f"**{clicked_hosp}** 상세 분석")
                    target_df = df[df['병원명'].astype(str).str.strip() == clicked_hosp]
                    
                    if not target_df.empty:
                        # 1. 핵심 KPI
                        st.metric("총 처방금액", f"{int(target_df['처방금액'].sum()):,} 원")
                        rep_name = ', '.join(target_df['성명'].dropna().unique())
                        st.caption(f"**담당자:** {rep_name}")
                        st.divider()
                        
                        # 2. 벤치마크 분석 (지점 평균 대비)
                        st.markdown("##### 📊 지표 밸런스 (지점 평균 대비 %)")
                        st.caption("소속 지점 평균을 100%로 잡았을 때의 상대적 수준입니다.")
                        
                        target_branch = target_df['지점'].iloc[0] if '지점' in target_df.columns else None
                        avg_kpi = target_df[['HIR_Raw', 'RTR_Raw', 'PHR_Raw', 'PI_Raw']].mean().fillna(0)
                        
                        if target_branch:
                            branch_avg = df[df['지점'] == target_branch][['HIR_Raw', 'RTR_Raw', 'PHR_Raw', 'PI_Raw']].mean().replace(0, 1)
                            bench_values = [
                                (avg_kpi['HIR_Raw'] / branch_avg['HIR_Raw']) * 100,
                                (avg_kpi['RTR_Raw'] / branch_avg['RTR_Raw']) * 100,
                                (avg_kpi['PHR_Raw'] / branch_avg['PHR_Raw']) * 100,
                                (avg_kpi['PI_Raw'] / branch_avg['PI_Raw']) * 100
                            ]
                        else:
                            bench_values = [100, 100, 100, 100]
                            
                        categories = ['활동질(HIR)', '관계온도(RTR)', '파이프라인(PHR)', '성과지수(PI)']
                        
                        import plotly.express as px
                        fig = px.bar(
                            x=bench_values,
                            y=categories,
                            orientation='h',
                            text=[f"{v:.1f}%" for v in bench_values],
                            color=bench_values,
                            color_continuous_scale='RdYlGn', # 낮은값 빨강, 높은값 초록
                            range_x=[0, max(200, max(bench_values)*1.1)]
                        )
                        fig.add_vline(x=100, line_dash="dash", line_color="gray", annotation_text="지점 평균")
                        fig.update_layout(
                            xaxis_title="", yaxis_title="",
                            height=250, margin=dict(l=20, r=20, t=20, b=20),
                            coloraxis_showscale=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.divider()

                        # 3. [신규] 활동-실적 인사이트 진단
                        st.markdown("##### 💡 활동-실적 전략 인사이트")
                        
                        # 효율성 계산: 총 실적 / 활동 횟수
                        total_act = len(target_df.dropna(subset=['activities']))
                        total_sales = target_df['처방금액'].sum()
                        efficiency = total_sales / (total_act if total_act > 0 else 1)
                        
                        # 지점 평균 효율성
                        branch_total_act = len(df[df['지점'] == target_branch].dropna(subset=['activities']))
                        branch_total_sales = df[df['지점'] == target_branch]['처방금액'].sum()
                        branch_efficiency = branch_total_sales / (branch_total_act if branch_total_act > 0 else 1)
                        
                        insight_cols = st.columns(2)
                        with insight_cols[0]:
                            st.metric("활동 효율성", f"{efficiency / 1000:,.1f}k", 
                                      delta=f"{(efficiency/branch_efficiency-1)*100:.1f}%" if branch_efficiency>0 else None,
                                      help="방문 1회당 발생하는 처방금액 효율 (지점 평균 대비)")
                        
                        # 텍스트 오토-가이드 생성 (Rule-based)
                        advice = ""
                        if (efficiency > branch_efficiency) and (avg_kpi['PI_Raw'] > 100):
                            advice = "✅ **고효율 관리 모델:** 적은 방문으로도 높은 성과를 내고 있습니다. 현재의 활동 질(HIR)을 유지하며 리소스 여력을 경쟁 병원 침투에 활용하십시오."
                        elif (efficiency < branch_efficiency) and (avg_kpi['HIR_Raw'] > branch_avg['HIR_Raw']):
                            advice = "⚠️ **활동 과잉 신호:** 활동의 질은 좋으나 실적 전환율이 낮습니다. 단순 방문보다는 처방 의사결정권자와의 관계 심화(RTR)가 필요합니다."
                        elif (avg_kpi['PHR_Raw'] < branch_avg['PHR_Raw']):
                            advice = "🚨 **파이프라인 경고:** 미래 성과 지표(PHR)가 낮습니다. 현재 실적 유지에만 급급할 수 있으니, 신규 품목 제안 활동을 즉시 강화하십시오."
                        else:
                            advice = "📈 **안정적 성장세:** 지점 평균 수준의 밸런스를 유지하고 있습니다. 주기적인 방문 간격을 유지하며 이탈 방지에 주력하십시오."
                        
                        st.info(advice)
                        st.divider()
                        
                        # 4. 최근 활동 타임라인 (하단 이동)
                        st.markdown("##### 📅 최근 리얼 활동 로그")
                        act_df = target_df.dropna(subset=['activities']).sort_values('날짜', ascending=False).head(3)
                        if not act_df.empty:
                            for _, act_row in act_df.iterrows():
                                dt_str = act_row['날짜'].strftime('%m/%d') if pd.notnull(act_row['날짜']) else 'N/A'
                                st.markdown(f"📌 `{dt_str}` | {act_row['activities']}")
                        else:
                            st.caption("기록된 활동이 없습니다.")
                    else:
                        st.warning("선택된 병원의 상세 데이터가 존재하지 않습니다.")
                else:
                    st.info("☝️ 왼쪽 지도에서 반짝이는 초록색 마커를 클릭하시면 이곳에 담당자의 상세 분석 결과가 나타납니다.")
        else:
            st.warning(f"설정된 맵 파일({map_path})을 아직 찾을 수 없습니다. (먼저 지도를 생성해주세요)")
