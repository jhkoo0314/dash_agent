# SFE Master Sandbox (V13.1)

Sales / Target / CRM 데이터를 통합해 아래 결과물을 생성합니다.

- Streamlit 기반 SFE 분석 샌드박스
- 최종 전략 리포트 HTML
- 전국병원 지도(Spatial Preview) HTML

## 1. 프로젝트 구성

- 대시보드 엔트리: `scripts/sfe_sandbox.py`
- 리포트 엔진: `scripts/report_builder_v12.py`
- 지도 데이터 빌더: `scripts/map_data_builder.py`
- 지도 탭 UI: `scripts/hospital_map_tab.py`
- 전략 리포트 템플릿: `templates/report_template.html`
- 지도 템플릿(기준 템플릿): `templates/spatial_preview_template.html`
- 컬럼 매핑 설정: `config/mapping.json`

## 2. 기술 스택

- Python
- streamlit
- pandas, numpy
- plotly
- scikit-learn
- jinja2
- openpyxl

설치:

```bash
pip install -r requirements.txt
```

## 3. 디렉터리 가이드

- `data/`
  - `data/sales/`: 실적 데이터
  - `data/targets/`: 목표 데이터
  - `data/crm/`: 활동 데이터
  - `data/logic/`: 좌표 조인 데이터(`hospital_assignment_data_v2.xlsx`)
- `scripts/`: 샌드박스/리포트/지도 빌드 로직
- `templates/`: HTML 템플릿
- `config/`: 매핑/설정 파일
- `docs/`: 기획/구현 문서
- `output/`: 결과물
  - `output/processed_data/standardized_sales_*.csv`
  - `output/processed_data/map_master_*.csv`
  - `output/Strategic_Full_Dashboard_*.html`
  - `output/Spatial_Preview_*.html`

## 4. 실행 방법

### 4.1 샌드박스 실행

```bash
streamlit run scripts/sfe_sandbox.py
```

### 4.2 전략 리포트 단독 생성

```bash
python scripts/report_builder_v12.py
```

## 5. 지도 생성 흐름 (샌드박스 탭)

전국병원 지도 탭은 2단계로 동작합니다.

1. **1단계 맵데이터 빌더 실행**
   - `map_data_builder.build_map_master_csv()` 호출
   - 4개 소스(활동/목표/실적/좌표)에서 필수 컬럼만 병합
   - 결과 CSV 생성:
     - `output/processed_data/map_master_YYMMDD(.n).csv`
   - 생성된 CSV 미리보기 및 다운로드 가능

2. **2단계 최종 HTML 생성**
   - `map_data_builder.build_spatial_preview_html_from_csv()` 호출
   - 기준 템플릿(`templates/spatial_preview_template.html`)에
     `__INITIAL_MARKERS__`, `__INITIAL_ROUTES__`를 주입
   - 결과 HTML 생성:
     - `output/Spatial_Preview_YYMMDD_HHMMSS(.n).html`

핵심:
- 지도는 템플릿 기반 주입 방식으로 생성됩니다.
- 데이터가 바뀌면 템플릿은 유지하고 payload만 교체합니다.

## 6. 지도용 병합 컬럼(요약)

- CRM: `활동일자`, `담당자명`, `요양기관명`
- TARGET: `요양기관명`, `목표월`, `목표금액`
- SALES: `요양기관명`, `목표월`, `실적금액`
- COORD: `요양기관명`, `경도`, `위도`

조인/정렬 규칙:
- 조인 키: `요양기관명` 정규화 키 + `활동월(활동일자에서 파생)`/`목표월`
- 이동 순서: `활동일자` + `source_row_no`(입력순 보존)

## 7. 파일명 규칙

- 표준화 매출 CSV:
  - `output/processed_data/standardized_sales_YYMMDD(.n).csv`
- 지도 마스터 CSV:
  - `output/processed_data/map_master_YYMMDD(.n).csv`
- 전략 리포트 HTML:
  - `output/Strategic_Full_Dashboard_YYMMDD(.n).html`
- 지도 미리보기 HTML:
  - `output/Spatial_Preview_YYMMDD_HHMMSS(.n).html`

## 8. 빠른 검증 체크리스트

- `streamlit run scripts/sfe_sandbox.py` 실행 성공
- STEP 1(마스터 로직) 결과 CSV 생성 성공
- 지도 탭 1단계 `map_master_*.csv` 생성 및 미리보기 성공
- 지도 탭 2단계 `Spatial_Preview_*.html` 생성 성공
- `python scripts/report_builder_v12.py` 생성 성공

## 9. 주의사항

- `data/`, `output/` 사용자 데이터는 삭제/초기화하지 않습니다.
- `config/mapping.json`은 기존 호환성을 유지합니다.
- 한글 포함 파일 편집/검증은 UTF-8 기준으로 처리합니다.
