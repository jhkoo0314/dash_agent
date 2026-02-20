import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import os
import glob

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
        "지점": ["지점", "Branch"], "성명": ["성명", "Rep"], "품목": ["품목", "Product"],
        "처방금액": ["처방금액", "Amount"], "목표금액": ["목표금액", "Target"],
        "월": ["월", "Month"], "activities": ["activities", "활동"], "segment": ["segment", "규모"]
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
            csvs = glob.glob(os.path.join(d, "*.csv"))
            available_files.extend(csvs)
    
    st.info(f"🔍 시스템이 {len(available_files)}개의 분석 가능한 파일을 찾았습니다.")
    
    selected_files = st.multiselect(
        "분석에 포함할 파일을 선택하세요", 
        options=available_files,
        default=available_files[:1] if available_files else [],
        help="data 폴더 내의 파일들이 자동으로 표시됩니다."
    )
    
    # 추가 업로드 기능 유지
    uploaded_files = st.file_uploader("그 외 추가로 업로드할 파일이 있다면 선택하세요", type="csv", accept_multiple_files=True)
    
    all_data_sources = selected_files + (uploaded_files if uploaded_files else [])
    
    if all_data_sources:
        # 파일 통합 로직
        df_list = []
        for f in all_data_sources:
            if isinstance(f, str): # 경로 문자열인 경우 (자동 탐색)
                df_list.append(pd.read_csv(f))
            else: # 업로드된 파일 객체인 경우
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
                    mapping_pairs = [("지점", m_br), ("성명", m_rep), ("품목", m_pd), ("처방금액", m_val), ("activities", m_act), ("날짜", m_dt), ("segment", m_seg)]
                    for key, val in mapping_pairs:
                        if key in mapping_config and val not in mapping_config[key]:
                            mapping_config[key].append(val)
                            updated = True
                    if updated:
                        save_mapping_config(mapping_config)
                        st.toast("🧠 시스템이 새로운 컬럼 매핑을 학습했습니다!", icon="⚡")

                # 컬럼명 표준화 (안전한 매핑)
                rename_map = {
                    m_br: '지점', m_rep: '성명', m_pd: '품목',
                    m_val: '처방금액', m_act: 'activities', 
                    m_dt: '날짜', m_seg: 'segment'
                }
                
                # 중복된 매핑 제거 (동일한 원본 컬럼이 다른 이름으로 두 번 매핑될 때 마지막 것 유지)
                df_std = raw_df.copy()
                df_std = df_std.rename(columns=rename_map)
                
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
                    df_std['처방수량'] = (df_std['처방금액'] / 1000).astype(int)
                
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
    st.info("📦 **리포트 빌더 전용 표준 파일 생성**")
    final_cols = ['지점', '성명', '품목', '처방금액', '처방수량', 'activities', 'segment', '날짜', 'HIR_Raw', 'RTR_Raw', 'PHR_Raw']
    export_df = df[final_cols]
    
    csv_out = export_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 리포트 빌더용 표준 파일(standardized_sales.csv) 다운로드",
        data=csv_out,
        file_name="standardized_sales.csv",
        mime="text/csv",
        help="이 파일을 다운로드하여 sfe_report_builder.py가 있는 폴더에 넣으세요."
    )
    
    st.dataframe(export_df.head(20))