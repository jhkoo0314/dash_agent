# SFE(Sales Force Effectiveness) 리포트 시스템

이 프로젝트는 제약 실적 데이터와 목표를 분석하여 시각화된 대시보드 리포트를 생성하는 시스템입니다.

## 📁 폴더 구조 (Project Structure)

```text
c:/agent_b/
├── data/                    # 모든 데이터 관련 파일
│   ├── sales/               # 실적(Sales) 데이터 (예: standardized_sales.csv)
│   ├── targets/             # KPI 목표(Target) 데이터 (예: target.csv)
│   ├── crm/                 # CRM 데이터 (고객 정보, 활동 로그 등)
│   └── logic/               # 비즈니스 로직 및 마스터 가중치 (SFE_Master_Logic.xlsx)
├── scripts/                 # 실행 스크립트
│   └── report_builder_v12.py # 메인 리포트 생성 엔진
├── templates/               # 디자인 템플릿
│   └── report_template.html # 리포트 HTML 뼈대
├── output/                  # 생성된 결과물 및 가공 데이터
│   ├── processed_data/      # 샌드박스에서 가공된 표준 데이터 (standardized_sales.csv)
│   └── Strategic_Full_Dashboard.html # 최종 결과 리포트
└── README.md                # 프로젝트 안내 가이드
```

## 🚀 사용 방법 (Usage)

### 1. 데이터 준비 및 가공 (Sandbox)

- **실적 데이터**: 로우데이터(CSV)를 `data/sales/` 폴더에 넣습니다.
- **목표 데이터**: 목표 CSV 파일을 `data/targets/` 폴더에 넣습니다.
- **샌드박스 실행**: 터미널에서 다음 명령어를 입력합니다.
  ```bash
  streamlit run scripts/sfe_sandbox.py
  ```
- 가공된 결과물인 `standardized_sales.csv`는 자동으로 `output/processed_data/` 폴더에 저장됩니다.

### 2. 리포트 생성 (Report Builder)

- 터미널에서 다음 명령어를 실행합니다:
  ```bash
  python scripts/report_builder_v12.py
  ```
- 시스템은 `output/processed_data/standardized_sales.csv`를 최우선으로 읽고, `data/targets/` 폴더의 목표 데이터를 결합하여 리포트를 생성합니다.

### 3. 결과 확인

- 실행이 완료되면 `output/` 폴더 안에 `Strategic_Full_Dashboard.html` 파일이 생성됩니다.
- 해당 파일을 브라우저(Chrome, Edge 등)로 열어 대시보드를 확인하십시오.

## 🛠️ 주요 기능

- **자동 파일 검색**: 폴더 내의 최신 실적 및 목표 파일을 자동으로 찾아 로드합니다.
- **T-Score 연산**: 복잡한 성과 지표를 표준화된 점수(0~100)로 변환합니다.
- **AI 분석 시뮬레이션**: RandomForest 기반 중요도 분석 및 보정 상관계수를 제공합니다.
- **반응형 대시보드**: 모바일 및 데스크탑에서 최적화된 시각화 리포트를 제공합니다.
