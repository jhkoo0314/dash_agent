# Map Engine Stage Spec

## 목적
맵빌더 실행 단계를 표준화하고 단계별 검증/로그 기준을 정의한다.

## M1 Load
- 입력: 시나리오 config, 입력 경로
- 출력: 도메인별 Raw DataFrame
- 검증: 파일 존재, 필수 컬럼 존재

## M2 Mapping/Normalize
- 입력: Raw + mapping rules
- 출력: 표준 컬럼 DataFrame
- 검증: 날짜/월 파싱률, 타입 변환 성공률

## M3 Keys/Join
- 입력: 표준 DataFrame + join keys
- 출력: `map_master`
- 검증: null/중복/row 증가율/미매핑률

## M4 Map Metrics
- 입력: `map_master`
- 출력: 시퀀스/거리/방문수 포함 중간 DataFrame
- 검증: 좌표 범위, 거리 sanity check

## M5 Payload
- 입력: 중간 DataFrame
- 출력: `markers`, `routes`
- 검증: 필수 필드, JSON 직렬화 가능 여부

## M6 Render
- 입력: payload + template
- 출력: 최종 HTML
- 검증: placeholder 치환 성공, 파일 생성 성공

## 오류 코드 예시
- `MAP_LOAD_FAIL`
- `MAP_SCHEMA_FAIL`
- `MAP_JOIN_FAIL`
- `MAP_COORD_QUALITY_FAIL`
- `MAP_RENDER_FAIL`

## 로그 필드 최소 기준
- `run_id`
- `scenario_name`
- `stage`
- `status`
- `row_count`
- `elapsed_ms`
- `error_code`
- `error_message`

