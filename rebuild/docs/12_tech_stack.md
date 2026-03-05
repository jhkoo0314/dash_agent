# 기술 스택 문서 (Rebuild)

## 1. 목적
실무 재구축 기준에서 "공통 엔진 1개 + 시나리오 config N개 + 템플릿 N개" 구조를 안정적으로 운영하기 위한 기술 스택 표준을 정의한다.

## 2. 핵심 원칙
- 데이터 처리 로직은 엔진에 집중한다.
- 시나리오 차이는 config와 template로 분리한다.
- 조인/집계는 ID 기반으로 수행한다.
- 실행 이력과 품질 검증 결과를 반드시 남긴다.

## 3. 언어 및 런타임
- Python 3.10+

## 4. 주요 라이브러리
- Streamlit: 작업 콘솔(UI)
- pandas, numpy: 로딩/정제/집계
- plotly: 시각화
- scikit-learn: 필요 시 분석/보조 모델링
- jinja2: HTML 보고서 렌더링
- openpyxl: Excel 입출력

## 5. 모듈별 권장 기술
- Loader: pandas + openpyxl
- Mapping/Standardization: pandas + json/yaml config
- Key/Join Validation: pandas + 사용자 정의 검증 함수
- Metrics: pandas groupby/agg
- Payload Builder: pandas -> dict/json 직렬화
- Renderer: jinja2 template
- Console App: streamlit

## 6. 디렉터리 표준
- config/
  - mapping.json
  - scenarios/*.yaml
  - schema/*.yaml
- scripts/
  - engine/*.py
  - apps/streamlit_app.py
- templates/
  - base/*
  - monthly_report.html
  - quarterly_report.html
  - business_report.html
- output/
  - processed/
  - reports/
  - logs/

## 7. 데이터/인코딩 표준
- 인코딩: UTF-8 고정
- 시간키: `metric_month=YYYY-MM`
- 조인키: 이름 기반 금지, ID 기반 강제
- 한글 검증 전 환경변수: `PYTHONIOENCODING=utf-8`

## 8. 품질/운영 기준
- 조인 전: null 비율, 중복, grain 적합성 검증
- 조인 후: 행 수 증가율 임계치 검증
- 실패 시: Fail-Fast 중단 + 단계/원인 로그 기록
- 산출물: 날짜 기반 파일명 + 충돌회피 suffix 유지

## 9. 실행 커맨드
- Dashboard(Console): `streamlit run scripts/sfe_sandbox.py`
- Report Build: `python scripts/report_builder_v12.py`

## 10. 확장 가이드
- 신규 시나리오: `config/scenarios/*.yaml` + `templates/*` 추가
- 신규 도메인: `data/{domain}` + `config/schema/domains/{domain}.yaml` 추가
- 엔진 코드 복제 금지
