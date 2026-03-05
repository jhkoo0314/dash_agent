# 06 Engine Stage Spec

## 목적
`load -> mapping -> keys/join -> metrics -> payload -> render` 단계별 입출력/검증 포맷을 정의한다.

## Stage 1: Load
- 입력: 시나리오 config, 파일 경로 패턴
- 출력: 도메인별 Raw DataFrame
- 검증: 파일 존재, 필수 시트/컬럼 존재

## Stage 2: Mapping
- 입력: Raw DataFrame, `config/mapping.json`
- 출력: 표준 컬럼 DataFrame
- 검증: 표준 컬럼 매핑률, 타입 변환 성공률

## Stage 3: Keys/Join
- 입력: 표준 DataFrame, grain/join_keys
- 출력: 통합 Master DataFrame
- 검증: null/중복/행증가율, 조인 성공률

## Stage 4: Metrics
- 입력: Master DataFrame, metrics 규칙
- 출력: 지표 DataFrame
- 검증: 지표 결측률, 값 범위 sanity check

## Stage 5: Payload
- 입력: 지표 DataFrame, 보고서 메타데이터
- 출력: Template payload(dict/json)
- 검증: 템플릿 필수 필드 누락 여부

## Stage 6: Render
- 입력: payload, HTML template
- 출력: HTML report 파일
- 검증: 파일 생성 여부, 렌더 오류 여부

## 로그 포맷 최소 기준
- run_id
- scenario_name
- stage
- status
- row_count/elapsed_ms
- error_message (실패 시)
