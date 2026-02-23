# SFE Master Sandbox (V13.1)

Sales/Target/CRM 데이터를 통합해 다음 결과물을 만드는 프로젝트입니다.

- Streamlit 기반 인터랙티브 SFE 분석 대시보드
- 최종 HTML 전략 리포트

## 1. 프로젝트 구성

- 대시보드 진입점: `scripts/sfe_sandbox.py`
- 리포트 엔진: `scripts/report_builder_v12.py`
- 리포트 템플릿: `templates/report_template.html`
- 지도 컴포넌트: `scripts/map_component/index.html`, `hospital_map.html`
- 컬럼 매핑 설정: `config/mapping.json`

## 2. 기술 스택

- Python
- streamlit
- pandas, numpy
- plotly
- scikit-learn
- jinja2
- openpyxl

의존성 설치:

```bash
pip install -r requirements.txt
```

## 3. 디렉터리 가이드

- `data/`: 원천 데이터
- `data/sales/`: 실적 파일
- `data/targets/`: 목표 파일
- `data/crm/`: CRM 활동 파일
- `scripts/`: 대시보드/리포트 로직
- `templates/`: HTML 템플릿
- `config/`: 매핑/설정 파일
- `docs/`: 설계/메모 문서
- `output/`: 생성 결과물
- `output/processed_data/standardized_sales_*.csv`
- `output/Strategic_Full_Dashboard_*.html`

## 4. 실행 방법

### 4.1 대시보드 실행

```bash
streamlit run scripts/sfe_sandbox.py
```

대시보드 주요 흐름:

1. `data/sales`, `data/targets`, `data/crm` 파일 자동 탐색 + 수동 업로드
2. 컬럼 매핑 확인/수정 (`config/mapping.json` 기반 별칭 자동 추천)
3. 6개 핵심 지표 로직 적용 후 표준화 파일 저장
4. 바로 최종 HTML 리포트 생성 버튼으로 연계
5. 병원 지도(Map Deep Dive)에서 클릭 기반 상세 분석 확인

### 4.2 최종 리포트 생성 (단독 실행)

```bash
python scripts/report_builder_v12.py
```

리포트 엔진 동작:

- 우선순위 1: `output/processed_data/standardized_sales_*.csv`
- 우선순위 2: `output/processed_data/standardized_sales.csv`
- 우선순위 3: `data/sales/` 원본 파일 fallback
- 목표 데이터는 `data/targets/`에서 자동 탐색
- 표준화 실적 파일을 사용하면 CRM 병합은 스킵
- 결과는 `output/Strategic_Full_Dashboard_YYMMDD(.n).html` 형태로 저장

## 5. 산출물 규칙

- 표준화 CSV: `output/processed_data/standardized_sales_YYMMDD(.n).csv`
- 최종 HTML: `output/Strategic_Full_Dashboard_YYMMDD(.n).html`
- 동일 날짜 파일이 있으면 `(1)`, `(2)` 식으로 충돌 회피

## 6. 컬럼 매핑/호환성

- 매핑 사전 파일: `config/mapping.json`
- 대시보드에서 사용자가 수정한 매핑을 학습(별칭 추가) 가능
- 리포트 엔진은 동일 매핑 사전을 재사용해 자동 정규화
- 키 컬럼 예시: `지점`, `성명`, `병원명`, `품목`, `처방금액`, `목표금액`, `월`, `activities`, `segment`, `날짜`

## 7. 빠른 검증 체크리스트

- `streamlit run scripts/sfe_sandbox.py` 실행 성공
- 대시보드에서 표준화 CSV 저장 성공
- `python scripts/report_builder_v12.py` 실행 성공
- `output/`에 최신 HTML 리포트 생성 확인
- `config/mapping.json` 기반 자동 매핑 동작 확인

## 8. 주의 사항

- `data/`, `output/` 내 사용자 데이터는 삭제/초기화하지 마세요.
- 매핑 로직 수정 시 `config/mapping.json`의 기존 키 호환성을 유지하세요.
- 콘솔에서 한글이 깨질 수 있으나, 파일 편집/저장은 UTF-8 기준으로 유지하세요.
