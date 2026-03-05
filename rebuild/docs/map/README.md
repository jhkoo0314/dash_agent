# Map Builder Rebuild Docs

이 폴더는 맵빌더 재구축을 위한 전용 문서 묶음입니다.
목표는 지도 생성 파이프라인을 리빌드 표준(데이터 계약, Fail-Fast, 구조화 로그, 회귀 테스트)에 맞춰 구현 가능하게 만드는 것입니다.

## 문서 구성
- `01_map_prd.md`: 제품 요구사항/범위/KPI/승인 기준
- `02_map_data_contract.md`: 입력/출력 데이터 계약
- `03_map_join_grain_policy.md`: 조인키/그레인/임계치
- `04_map_payload_schema.md`: map_master/markers/routes 스키마
- `05_map_scenario_spec.md`: `map_spatial_preview` 시나리오 YAML
- `06_map_engine_stage_spec.md`: M1~M6 단계 명세
- `07_map_quality_checklist.md`: 실행 전/중/후 품질 점검
- `08_map_runbook_release.md`: 운영 절차/장애 대응/릴리즈
- `09_map_security_governance.md`: 보안/권한/감사 기준
- `10_map_test_plan.md`: 단위/통합/회귀 테스트 계획

## 구현 순서(권장)
1. `01` + `02`로 요구사항/계약 고정
2. `03` + `05`로 조인 정책/시나리오 고정
3. `06` + `04`로 엔진/페이로드 구현
4. `07` + `10`으로 품질 게이트/테스트 자동화
5. `08` + `09`로 운영/보안 체계 마감

## 필수 원칙
- ID 기반 조인 우선, 이름 조인은 fallback으로만 사용
- quality gate는 config 기반으로 관리(하드코딩 금지)
- 실패는 즉시 중단(Fail-Fast)하고 run_id 로그에 기록
- UTF-8 고정, 원본/기존 산출물 삭제 금지

