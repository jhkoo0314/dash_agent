# SFE Rebuild Workspace

이 폴더는 기존 프로젝트를 즉시 수정하는 공간이 아니라, 재구축(Rebuild) 기준과 실행 설계를 운영하기 위한 문서 중심 워크스페이스입니다.

## 목적
- 공통 엔진 1개로 다중 시나리오(`monthly`, `quarterly`, `business`)를 운영한다.
- 코드 분기 대신 `config + template` 중심 확장 구조를 확립한다.
- Fail-Fast 검증과 구조화 로그로 결과 재현성과 추적성을 확보한다.

## 문서 구성
- `docs/01_architecture.md`: 목표 아키텍처 및 책임 분리
- `docs/02_data_contract.md`: 표준 컬럼/필수 키/타입/포맷 계약
- `docs/03_mapping_policy.md`: `mapping.json` 변경 정책 및 하위호환
- `docs/04_join_grain_policy.md`: grain/조인 규칙 및 Fail-Fast 기준
- `docs/05_scenario_spec_template.md`: 시나리오 YAML 템플릿
- `docs/06_engine_stage_spec.md`: 파이프라인 단계 명세
- `docs/07_runbook_ops.md`: 운영 절차 및 장애 대응
- `docs/08_quality_checklist.md`: 실행 전/중/후 품질 체크
- `docs/09_release_change_log.md`: 변경관리 및 롤백 원칙
- `docs/10_security_governance.md`: 보안/권한/감사 기준
- `docs/11_prd.md`: 제품 요구사항(PRD)
- `docs/12_tech_stack.md`: 기술 스택 표준
- `docs/13_rebuild_improvement_comparison.md`: As-Is vs To-Be 비교/로드맵
- `docs/14_map_builder_rebuild_spec.md`: 맵빌더 전용 재구축 기준

## Rebuild 핵심 원칙
1. 엔진은 1개만 유지한다.
2. 시나리오 차이는 `config/scenarios/*.yaml`과 템플릿으로만 분기한다.
3. Streamlit은 실행 콘솔 역할만 수행한다.
4. 조인은 ID 기반으로 수행하고 이름 조인은 금지한다.
5. 조인 전/후 검증 실패 시 즉시 중단(Fail-Fast)한다.
6. 실행 결과와 오류는 `run_id` 기준으로 구조화 로그를 남긴다.
7. UTF-8 인코딩을 고정하고 한글 깨짐 출력 기반 수정은 금지한다.
8. `data/`, `output/` 원본/산출물 무결성을 훼손하지 않는다.

## 목표 아키텍처(요약)
파이프라인 단계:
1. Load
2. Mapping
3. Keys/Join
4. Metrics
5. Payload
6. Render

권장 모듈 구조:
```text
scripts/
  engine/
    orchestrator.py
    loaders.py
    mapping.py
    keys.py
    metrics.py
    payload.py
    render.py
  apps/
    streamlit_app.py
config/
  mapping.json
  scenarios/*.yaml
  schema/*.yaml
templates/
  *.html
output/
  processed/
  reports/
  logs/
```

## 데이터 계약 요약
- 표준 컬럼명: 영문 `snake_case`
- 필수 키: `hospital_id`, `product_id`, `branch_id`, `metric_month`
- 시간키 포맷: `metric_month=YYYY-MM`
- 타입 원칙:
  - ID: string
  - 금액/수치: numeric
  - 텍스트: UTF-8
- 위반 처리: 계약 위반 단계 즉시 중단 + 위반 샘플 로그 기록

## 품질 게이트(필수)
- 조인 전:
  - 필수 키 null 비율
  - 조인키 중복
  - grain 적합성
- 조인 후:
  - row 증가율 이상치
  - 조인 성공률
- 지표 계산 후:
  - 결측률
  - 허용 범위(sanity check)

## 운영 원칙
- 일상 흐름:
  1. 입력 반입
  2. 시나리오 선택
  3. 실행
  4. 로그 확인
  5. 결과 검토
  6. 배포
- 재실행 기준:
  - 매핑/키/포맷 수정 후 동일 조건으로 재실행
  - 재현성 확인(동일 입력 + 동일 config)

## 릴리즈/변경관리
- 변경 유형: `config | template | engine`
- 각 변경은 반드시 아래를 기록:
  - 변경 사유
  - 영향 시나리오/도메인
  - 검증 결과
  - 롤백 계획

## 보안/거버넌스
- 최소 권한 원칙(Role-based)
- 실행/변경 이력 감사 가능 상태 유지
- 민감정보 마스킹 및 외부 반출 최소화

## 우선순위 로드맵(요약)
- Phase 0: 데이터 계약 강제, 조인 Fail-Fast, 구조화 로그
- Phase 1: 엔진 모듈 분리, 시나리오/템플릿 분리
- Phase 2: 테스트 자동화, 지표 sanity 자동화
- Phase 3: 릴리즈/감사/운영 KPI 고도화

## 비범위(Out of Scope)
- 완전 자율 에이전트 의사결정
- Streamlit 내부의 복잡 집계 직접 구현
- 원본 데이터 삭제/초기화 자동화

## 참고
리빌드 작업 시 상세 기준은 반드시 `docs/` 원문을 우선 참조합니다.
