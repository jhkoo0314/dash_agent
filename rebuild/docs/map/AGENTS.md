# AGENTS.md (Map Builder Rebuild)

## 목적
이 파일은 `rebuild/docs/map` 문서 기준으로 맵빌더를 구현/운영할 때 따라야 할 에이전트 작업 규칙을 정의한다.

## 우선 참조 순서
1. `01_map_prd.md`
2. `02_map_data_contract.md`
3. `03_map_join_grain_policy.md`
4. `05_map_scenario_spec.md`
5. `06_map_engine_stage_spec.md`
6. `04_map_payload_schema.md`
7. `07_map_quality_checklist.md`
8. `10_map_test_plan.md`
9. `08_map_runbook_release.md`
10. `09_map_security_governance.md`

## 구현 규칙
- 엔진 단계는 `M1~M6`를 유지한다.
- 조인은 ID 키 우선, fallback 이름 조인 사용 시 비율을 로그에 기록한다.
- 품질 임계치는 시나리오 config(`quality_gates`)에서만 관리한다.
- 템플릿은 렌더링 계층만 담당한다(비즈니스 로직 금지).

## 품질/실패 처리 규칙
- 아래 중 하나라도 위반하면 Fail-Fast 중단:
  - 키 null 비율 임계치 초과
  - 조인키 중복
  - row 증가율 이상치
  - 좌표 결측/범위 이탈 임계치 초과
  - 병원키 미매핑률 임계치 초과
- 실패 시 `run_id`, `stage`, `error_code`, `error_message`를 반드시 남긴다.

## 산출물 규칙
- `map_master_{date}(.n).csv`
- `Spatial_Preview_{date}_{time}(.n).html`
- 기존 산출물 덮어쓰기/삭제 금지

## 테스트 규칙
- 단위/통합/회귀 테스트를 모두 운영한다.
- 회귀 기준:
  - markers/routes count 편차: ±5%
  - `avg_route_km` 편차: ±10%

## 보안/인코딩 규칙
- UTF-8 고정
- 검증 전 `PYTHONIOENCODING=utf-8` 설정
- 원본 데이터(`data/`) 삭제/초기화 금지
- 위치/기관 데이터 반출 최소화

## 완료 기준(DoD)
- 동일 입력+동일 config 재현 가능
- 품질 게이트 및 Fail-Fast 동작 검증 완료
- 로그만으로 실패 원인 재현 가능
- 문서(`01~10`)와 구현 간 불일치 없음

