# 14 Map Builder Rebuild Spec

## 목적
전국병원 지도 빌더를 재구축할 때 필요한 아키텍처, 데이터 계약, 단계 명세, 품질 게이트, 운영 기준을 단일 문서로 정의한다.

## 범위
- 포함
  - 맵 마스터 CSV 생성(`map_master`)
  - Spatial Preview HTML 생성(`spatial_preview`)
  - 지도용 payload(`markers`, `routes`) 생성/검증
  - 지도 생성 실행 로그 및 품질 지표
- 제외
  - 신규 지오코딩 파이프라인 구축
  - 지도 라이브러리 프레임워크 교체 자체를 목표로 한 리빌드

## 현재 기준 구현(참조)
- 빌더: `scripts/map_data_builder.py`
- 탭 UI: `scripts/hospital_map_tab.py`
- 템플릿: `templates/spatial_preview_template.html`
- 산출물:
  - `output/processed_data/map_master_YYMMDD(.n).csv`
  - `output/Spatial_Preview_YYMMDD_HHMMSS(.n).html`

## 아키텍처 원칙
1. 지도 처리도 공통 엔진 단계(`load -> mapping -> keys/join -> metrics -> payload -> render`)를 따른다.
2. 지도 전용 로직은 `engine/map/*` 또는 동등 모듈로 분리한다.
3. Streamlit은 버튼/실행/다운로드/로그 조회만 담당한다.
4. 템플릿은 렌더링만 담당하고 데이터 처리 로직은 금지한다.

## 데이터 계약 (Map)
### 입력 도메인 및 필수 컬럼
- CRM
  - `activity_date`(원천 예: `활동일자`)
  - `rep_id` or `rep_name`(원천 예: `담당자ID`/`담당자명`)
  - `hospital_id` or `hospital_name`(원천 예: `병원ID`/`요양기관명`)
- SALES
  - `hospital_id` or `hospital_name`
  - `metric_month` (`YYYY-MM`, 원천 예: `목표월`)
  - `sales_amount`(원천 예: `실적금액`)
- TARGET
  - `hospital_id` or `hospital_name`
  - `metric_month`
  - `target_amount`(원천 예: `목표금액`)
- COORD
  - `hospital_id` or `hospital_name`
  - `lon`(원천 예: `경도`)
  - `lat`(원천 예: `위도`)

### 조인키/그레인
- 권장 마스터키: `hospital_id + metric_month`
- 병원명 fallback 사용 시 정규화 키 규칙을 문서화하고 로그에 fallback 비율 기록
- 기본 그레인
  - `map_master`: CRM 이벤트 row 기준 + 월 단위 실적/목표 결합
  - `markers`: `rep + metric_month + activity_date + hospital`
  - `routes`: `rep + metric_month + activity_date`

## 단계 명세 (Map-Specific)
### Stage M1: Load
- 파일 탐색/로드
- 검증:
  - 입력 파일 존재
  - 필수 컬럼 존재

### Stage M2: Mapping/Normalize
- 컬럼 표준화 + 타입 변환 + 시간키 통일(`YYYY-MM`)
- 검증:
  - `activity_date` 파싱 성공률
  - `metric_month` 파싱 성공률

### Stage M3: Keys/Join
- 병원키/월키 기준으로 SALES, TARGET, COORD 결합
- 검증(Fail-Fast):
  - 필수 키 null 비율 > 임계치 시 중단
  - 키 중복/충돌 발견 시 중단
  - 조인 후 row 증가율 이상치 시 중단

### Stage M4: Map Metrics
- 파생값 생성:
  - 방문 순서(`seq`)
  - 경로거리(`total_km`, haversine)
  - 방문수(`visits`)
- 검증:
  - 좌표 범위 sanity check (`lat [-90,90]`, `lon [-180,180]`)
  - 거리 음수/비정상값 방지

### Stage M5: Payload
- `markers`, `routes` JSON 생성
- 검증:
  - 필수 필드 누락 없음
  - JSON 직렬화 가능

### Stage M6: Render
- 템플릿 플레이스홀더 주입 후 HTML 출력
- 검증:
  - 플레이스홀더 치환 성공
  - 파일 생성 성공

## Fail-Fast 권장 임계치
- `activity_date` 파싱 실패율 > 5%: 중단
- 좌표 결측률 > 10%: 중단
- 좌표 범위 이탈률 > 1%: 중단
- 조인 후 row 증가율 > 120%: 중단
- 병원키 미매핑률 > 2%: 중단

임계치는 시나리오 config에서 관리하며 하드코딩 금지.

## 출력 규칙
- 맵 마스터 CSV:
  - `output/processed_data/map_master_{date}(.n).csv`
- 지도 HTML:
  - `output/Spatial_Preview_{date}_{time}(.n).html`
- 기존 산출물 보존(삭제/초기화 금지)

## 로그/관측성
- run_id 단위로 `output/logs/{run_id}.json` 저장
- 최소 필드:
  - `run_id`, `scenario_name`, `stage`, `status`, `row_count`, `elapsed_ms`, `error_code`, `error_message`
- Map 전용 메트릭:
  - `coord_missing_rate`
  - `coord_out_of_range_rate`
  - `hospital_unmatched_rate`
  - `route_count`
  - `avg_route_km`

## 테스트 기준
### 단위 테스트
- 키 정규화 함수
- 월 파싱 함수
- 거리 계산 함수(haversine)
- payload 생성 함수(markers/routes)

### 통합 테스트
- 샘플 입력 4도메인 -> `map_master` 생성 성공
- `map_master` -> `Spatial_Preview` 생성 성공
- 플레이스홀더 주입 실패 케이스 검증

### 회귀 테스트
- 고정 샘플셋 기준 마커/루트 개수 변화 감시
- 주요 지표(`route_count`, `avg_route_km`) 허용 편차 검증

## 운영 Runbook (Map)
1. 1단계 `map_master` 생성
2. 미리보기/결측률/좌표 품질 확인
3. 2단계 `spatial_preview` 생성
4. 실행 로그 저장 확인
5. 배포 전 품질 체크리스트 통과 확인

## 보안/인코딩
- UTF-8 고정
- `PYTHONIOENCODING=utf-8` 설정 후 검증
- 위치/기관 데이터 외부 반출 시 최소 범위 원칙 적용

## 완료 기준(DoD)
- 맵 생성 2단계가 동일 config로 재현 가능
- Fail-Fast 게이트가 좌표/조인 이상치를 차단
- 로그만으로 실패 원인 재현 가능
- 산출물 네이밍 규칙과 기존 호환성 유지

