# 01 Architecture

## 목적
공통 엔진, 시나리오 config, 템플릿을 분리해 확장성과 재현성을 확보한다.

## 아키텍처 원칙
- 엔진은 1개만 유지한다.
- 시나리오 차이는 config와 template로만 분기한다.
- UI(Streamlit)는 실행 콘솔 역할만 수행한다.
- 데이터 처리 로직은 템플릿에 넣지 않는다.

## 구성요소와 책임
- `engine/orchestrator`: 파이프라인 실행 제어
- `engine/loaders`: 입력 탐색/로딩
- `engine/mapping`: 컬럼 표준화/타입 정규화
- `engine/keys`: 키 생성, 키 검증, 조인
- `engine/metrics`: 지표 계산
- `engine/payload`: 템플릿 주입 데이터 구성
- `engine/render`: HTML 렌더 및 출력
- `apps/streamlit_app`: 업로드, 시나리오 선택, 실행/로그/다운로드

## 데이터 흐름
1. Input Load
2. Standardize
3. Key/Join Validate
4. Metrics Compute
5. Payload Build
6. Template Render
7. Output + Logs

## 확장 규칙
- 신규 시나리오: `config/scenarios/*.yaml` + `templates/*` 추가
- 신규 도메인: `data/{domain}` + `config/schema/domains/{domain}.yaml` 추가
- 엔진 코드 복제 금지
