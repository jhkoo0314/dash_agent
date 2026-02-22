# SFE (Sales Force Effectiveness) 리포트 시스템 (V13.1)

본 프로젝트는 제약 영업의 실적, 목표, CRM 활동 데이터를 통합 분석하여 다차원적인 관점(지점, 사원, 품목)에서 성과와 역량을 측정하는 시각화 대시보드 시스템입니다. **SFE Master Sandbox V13.1**을 통해 동적 매핑 및 AI 별명 학습을 지원하며, **Map Deep Dive** 기능을 통해 데이터와 지도 간의 양방향 탐색이 가능합니다.

## 📁 폴더 구조 (Project Structure)

```text
c:/agent_b/
├── data/                    # 모델 구동을 위한 기초 데이터 저장소
│   ├── sales/               # 영업 실적 (예: hospital_performance.xlsx)
│   ├── targets/             # KPI 지표 및 재무 목표 (예: hospital_monthly_targets.xlsx)
│   ├── crm/                 # CRM 활동 내역 (예: daily_crm_activity_2026.xlsx)
│   ├── public/              # 공공 데이터 (예: 심평원 병원정보서비스)
│   └── logic/               # 비즈니스 가중치 룰셋
├── scripts/                 # 메인 실행 스크립트 모음
│   ├── sfe_sandbox.py       # (UI) V13.1 데이터 통합, 스마트 매핑 및 딥다이브 플래시보드
│   ├── report_builder_v12.py# (Engine) 전략 분석, T-Score 표준화 및 HTML 리포트 생성
│   └── map_component/       # (Frontend) JS-Streamlit 브릿지가 포함된 커스텀 지도 컴포넌트
├── templates/               # 프론트엔드 디자인 & 뷰 템플릿
│   └── report_template.html # 최종 결과물의 HTML/JS 레이아웃 뼈대
├── output/                  # 생성된 결과물 적재 폴더
│   ├── processed_data/      # 샌드박스를 거친 정제/표준화 병합 데이터 (standardized_sales_*.csv)
│   └── Strategic_Full_Dashboard_*.html # 분석이 완료된 배포용 대시보드 리포트
├── config/                  # 설정 파일
│   └── mapping.json         # (AI 학습) 컬럼 별명 사전 및 매칭 환경 설정
├── hospital_map.html        # Folium으로 생성된 기초 병원 지도 자료
└── README.md                # (본 문서) 프로젝트 가이드
```

## 🚀 주요 기능 (Key Features)

### 1. SFE Agile Sandbox V13.1

- **AI 별명 학습 (Alias Learning):** 사용자가 한 번 매핑한 컬럼명(예: '요양기관명' → '병원명')을 기억하여 다음 분석 시 자동으로 매칭합니다.
- **스마트 활동 전파 (Smart Activity Propagation):** CRM 데이터와 실적 데이터의 키가 완벽하지 않아도, 지점/성명/품목/월 기반의 다중 키 매칭을 통해 활동명(activities)을 누락 없이 전파합니다.
- **동적 파라미터 튜닝:** 사이드바에서 6대 핵심 지표의 가중치와 감쇠 상수를 실시간으로 조절하며 시뮬레이션할 수 있습니다.

### 2. Map Deep Dive Analysis (📍 딥다이브 플래시보드)

- **양방향 통신 브릿지:** JS 지도에서 마크를 클릭하면 Streamlit 백엔드로 데이터가 즉시 전달되어 상세 분석 패널이 업데이트됩니다.
- **실시간 오버레이:** 기존 지도 위에 현재 실적 및 담당자 데이터를 동적으로 오버레이하여 시각화합니다.
- **전략 인사이트 진단:** 클릭한 병원의 효율성(실적/활동)을 지점 평균과 비교하여 '고효율 관리', '활동 과잉', '파이프라인 경고' 등의 자동 가이드를 제공합니다.

## 🚀 사용 방법 (Workflow)

### STEP 1. 데이터 통합 및 전략 주입 (SFE Sandbox)

1. 로컬 환경에서 Streamlit 앱을 실행합니다.
   ```bash
   streamlit run scripts/sfe_sandbox.py
   ```
2. **데이터 통합:** `data/` 폴더 내의 파일들을 선택하거나 직접 업로드하여 하나로 병합합니다.
3. **스마트 매핑:** 시스템이 추천하는 매핑을 확인하고 필요한 경우 수정합니다. (수정 시 향후 자동 매칭을 위해 학습됩니다.)
4. **마스터 로직 적용:** `[🚀 마스터 로직 적용]` 버튼을 클릭하여 `output/processed_data/`에 표준화된 데이터를 생성합니다.
5. **지도 탐색:** 하단의 지도 탭에서 병원을 클릭하여 **딥다이브 플래시보드**의 분석 결과를 확인합니다.

### STEP 2. 전략 리포트 빌드 (Report Builder)

1. 추출된 표준 CSV를 기반으로 최종 HTML 리포트를 생성합니다. (샌드박스 내 버튼 또는 직접 실행)
   ```bash
   python scripts/report_builder_v12.py
   ```
2. **T-Score 표준화:** 6대 지표를 변별력 높은 T-Score 체계(25배 가중치)로 환산하여 상대평가 리포트를 구성합니다.
3. **결과 확인:** `output/` 폴더에 생성된 `Strategic_Full_Dashboard_{Date}.html`을 웹 브라우저로 엽니다.

## 🧠 6대 핵심 전략 지표 (6 Key Metrics)

- **HIR (High-Impact Rate, 유효행동):** 높은 질적 가치를 지닌 활동(PT, 클로징 등) 비중
- **RTR (Relationship Temp, 관계온도):** 최근 방문 시점 기반의 시간 감쇠(λ) 모형 적용 핵심 지수
- **BCR (Behavior Consistency, 규칙성):** 영업 활동 빈도의 편차와 꾸준함 측정
- **PHR (Pipeline Health, 계획성):** 미래 실적 전환을 목적으로 하는 전략적 활동 비율
- **PI (Prescription Index, 난이도 보정):** 상급종합병원 등 진입 장벽이 높은 거래처 실적 우대 지수
- **FGR (Field Growth Rate, 성장률):** 처방수량(Q)과 약가(P)의 밸런스 성장 지표

## 🛠 기술 스펙 및 특징

- **Normalization:** 전사 평균을 기준으로 한 상대적 위치 파악을 위해 T-Score 및 레이더 차트 전용 스케일링 적용
- **Automation:** 날짜 형식(YYYY-MM-DD, YYYY-MM 등)에 관계없이 분기/월 단위 자동 파싱 지원
- **Interactive:** Folium 마커 세트와 Streamlit 간의 커스텀 컴포넌트 기반 데이터 연동
