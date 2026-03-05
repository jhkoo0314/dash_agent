# AGENTS.md (Rebuild Workspace)

## 목적
이 파일은 `C:\agent_b\rebuild` 영역에서 작업하는 에이전트/개발자가 따라야 할 실행 규칙을 정의합니다.
기준 문서는 `rebuild/docs/*.md`이며, 본 문서는 실무 적용용 운영 지침입니다.

## 작업 범위
- 이 폴더의 기본 목표는 "재구축 설계/정책/표준" 정립입니다.
- 기존 운영 코드의 즉시 대규모 변경보다, 정책을 코드로 강제할 수 있는 구조 설계를 우선합니다.
- 직접 코드를 작성할 때도 `docs`의 원칙을 위반하지 않아야 합니다.

## 우선 적용 문서 순서
1. `docs/13_rebuild_improvement_comparison.md` (우선순위/갭 기준)
2. `docs/11_prd.md` (요구사항/성공지표)
3. `docs/01_architecture.md` ~ `docs/06_engine_stage_spec.md` (구조/단계 명세)
4. `docs/07_runbook_ops.md` ~ `docs/10_security_governance.md` (운영/품질/보안)
5. `docs/12_tech_stack.md` (기술 표준)
6. `docs/14_map_builder_rebuild_spec.md` (맵빌더 전용 기준)

## 아키텍처 규칙
- 공통 엔진 1개 유지.
- 시나리오 차이는 `config/scenarios/*.yaml` + `templates/*`로 분리.
- UI(Streamlit)는 실행 트리거/로그 조회/다운로드만 담당.
- 템플릿에 데이터 처리 로직을 넣지 않는다.
- 엔진 코드 복제 금지.

## 데이터 계약 규칙
- 표준 컬럼은 영문 `snake_case`.
- 필수 키:
  - `hospital_id`
  - `product_id`
  - `branch_id` (필요 시)
  - `metric_month` (`YYYY-MM`)
- 조인/집계는 ID 기반으로 수행하며 이름 조인은 금지.
- 계약 위반 시 Fail-Fast 중단.

## 조인/그레인 검증 규칙
- 조인 전 필수 검증:
  - 키 null 비율
  - 키 중복
  - grain 정합성
- 조인 후 필수 검증:
  - row 증가율
  - 이상치 여부
- 임계치 초과 시 실행 중단 + 원인 로그 기록.

## 파이프라인 단계 규칙
고정 단계:
1. Load
2. Mapping
3. Keys/Join
4. Metrics
5. Payload
6. Render

단계별 최소 로그 필드:
- `run_id`
- `scenario_name`
- `stage`
- `status`
- `row_count`
- `elapsed_ms`
- `error_message`(실패 시)

## 매핑 정책 규칙
- `config/mapping.json` 하위호환 유지(기존 표준키 삭제 금지).
- 변경은 추가/수정/폐기로 분류하고 사유와 영향범위를 기록.
- 매핑 변경 시 샘플 회귀 검증을 수행.

## 테리토리 매핑(신규 필수 반영)
`docs/13` 기준으로 아래를 강제:
- 표준키 후보: `territory_id`, `territory_name`, `branch_id`, `branch_name`
- 우선순위: 코드 기반 매핑 우선, 이름은 보조.
- 미매핑/충돌률 임계치 관리 및 Fail-Fast.
- 월별 변경 영향 리포트가 가능해야 함.

## 품질/테스트 규칙
- 최소 테스트 기준:
  - stage 단위 단위 테스트
  - 시나리오 통합 테스트
  - 골든 스냅샷 회귀 테스트
- 품질 체크리스트(`docs/08`)를 실행 전/중/후 적용.
- 테스트 실패 상태에서 병합/배포 금지.

## 운영/릴리즈 규칙
- 변경 유형(`config/template/engine`)별 이력 관리.
- 각 변경에 대해 영향 분석/검증 결과/롤백 계획을 반드시 기록.
- 실패 시 장애 리포트(원인 분류 + 재실행 결과) 남김.

## 보안/거버넌스 규칙
- `data/`, `output/` 삭제/초기화 금지.
- 최소 권한 원칙, 승인 절차 준수.
- 실행/변경 이력은 감사 가능 형태로 보관.
- 민감정보는 마스킹 후 처리/공유.

## 인코딩 규칙
- UTF-8 고정.
- 한글 깨짐 콘솔 출력은 수정 근거로 사용 금지.
- Python 검증 전:
```powershell
$env:PYTHONIOENCODING='utf-8'
```
- 깨짐이 보이면 원본 파일(UTF-8)을 직접 확인하고 수정.

## 산출물/네이밍 규칙
- 날짜 기반 파일명 + 충돌 회피 suffix 유지.
- 기존 산출물 보존(덮어쓰기/삭제 지양).
- 실행 로그는 `output/logs/` 또는 동등 경로에 run_id 기준 저장.

## 작업 우선순위 (P0 -> P1)
- P0:
  - 데이터 계약 validator
  - 조인 Fail-Fast 게이트
  - run_id 구조화 로그
  - 인코딩/국문 안정성
- P1:
  - 매핑 변경관리 자동화
  - 지표 카탈로그/범위 검증
  - 릴리즈/감사 체계 고도화

## 금지사항
- 이름 기반 조인
- 엔진 코드 복제
- 템플릿 내 비즈니스 로직 삽입
- 근거 없는 하드코딩 확장
- 원본 데이터 파괴적 수정

## 완료 기준(DoD)
- 문서 정책과 코드/설정이 충돌하지 않음
- 동일 입력+동일 config에서 재현 가능한 결과 생성
- 실패 시 로그만으로 원인 재현 가능
- 하위호환성(mapping) 훼손 없음
