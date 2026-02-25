# 업무자동화 워크플로우 가이드 (Semi-Agent Architecture)

## 1) 문서 목적
이 문서는 실무에서 재사용 가능한 **업무자동화 시스템(세미 에이전트)** 을 설계/구축/운영하기 위한 표준 가이드다.
목적은 특정 프로젝트 재현이 아니라, 여러 업무 시나리오를 공통 엔진으로 운영하는 설계 원칙을 제공하는 것이다.

핵심 목표:
- 데이터 소스가 수시로 바뀌어도 코드 수정 최소화
- 도메인 폴더가 늘어나도(`data/{domain}`) 엔진 수정 없이 확장
- 결과물(월간/분기/업무보고 등) 수가 늘어나도 엔진 1개로 운영
- 자동화 품질(정확성, 재현성, 추적성) 확보

---

## 2) 시스템 철학: 완전자율 에이전트보다 "설정 주도 파이프라인"

### 왜 완전자율 에이전트가 어려운가
- 입력 스키마 변화가 잦고, 비즈니스 룰이 자주 바뀜
- 조인 실패/중복/누락은 작은 오차도 치명적
- 운영 현장에서는 "설명 가능성"과 "재현 가능성"이 더 중요

### 실무 정답 구조
- 고정 70%: 병합/집계/검증/출력 엔진
- 가변 30%: 시나리오 config + 템플릿 + 필터
- LLM/에이전트 역할: 매핑 후보 추천, 서술형 코멘트 생성 보조

한 줄 정의:
- **이 시스템은 완전 자율 에이전트가 아니라, 에이전트형 워크플로우를 적용한 설정 기반 자동화 엔진이다.**

---

## 3) 핵심 개념 정리

### 3.1 마스터 컬럼(표준 컬럼)
- 최종적으로 통일해 사용할 컬럼명 체계
- 예: `hospital_id`, `hospital_name`, `product_id`, `metric_month`, `sales_amount`

### 3.2 매핑 사전 (`config/mapping.json`)
- 다양한 원천 컬럼 별칭을 표준 컬럼으로 변환하기 위한 사전
- 역할: "이름 통일"
- 한계: 조인 규칙 자체를 해결하지는 않음

### 3.3 마스터키 vs 조인키
- 마스터키: 기준 엔터티 식별자 (예: `hospital_id`)
- 조인키: 테이블 결합에 쓰는 키 (예: `hospital_id + metric_month + product_id`)
- 실무에서는 같은 컬럼이 문맥에 따라 둘 다 될 수 있음

### 3.4 Grain(행 단위)
- 각 테이블의 1행이 의미하는 단위
- 조인 실패의 가장 흔한 원인은 **키가 아니라 grain 불일치**
- 예: Sales(병원+월) vs HIR Long(병원+월+활동유형)

### 3.5 Long vs Wide
- Long(세로형): 표준화/확장에 유리
- Wide(가로형): 최종 보고서/시각화에 유리
- 권장: "표준화는 Long, 산출은 Wide"

---

## 4) 표준 아키텍처

실무 표준 아키텍처는 `공통 엔진 1개 + 시나리오 config N개 + 템플릿 N개` 구조다.

### 4.1 공통 엔진 책임
1. 입력 로딩
- 소스 경로/패턴은 config에서 주입
- 파일 형식(CSV/XLSX) 공통 처리

2. 표준화
- 매핑 사전으로 원천 컬럼명을 표준 컬럼으로 변환
- 타입/날짜/단위 정규화

3. 키/조인
- 기준 grain에 맞춰 키 생성
- 키 검증(null/중복/폭증) 후 조인 수행

4. 지표 계산
- 시나리오별 지표 집계 규칙 적용
- 공통 파생지표(달성률, 갭 등) 계산

5. 렌더 준비
- 템플릿 주입용 payload(JSON/DF) 생성

6. 출력
- 템플릿 렌더 및 파일 저장
- 실행 로그/검증 결과 저장

### 4.2 시나리오 config 책임
1. 입력 정의
- 도메인 목록, 파일 경로/패턴, 우선순위

2. 비즈니스 규칙
- 필터, 조인키, grain, 집계 방식, 지표 목록

3. 출력 규칙
- 템플릿 파일, 출력 포맷, 파일명 규칙

4. 품질 기준
- 허용 결측률, 중복 허용치, 실패 조건

### 4.3 템플릿 책임
1. 표현 계층 분리
- 데이터 처리 로직 금지
- payload 표시/레이아웃만 담당

2. 시나리오별 분기
- 월간/분기/업무보고 템플릿 독립 관리
- 공통 컴포넌트(header/chart/table)는 재사용

### 4.4 아키텍처 원칙
- 엔진 코드는 복제하지 않는다.
- 시나리오 차이는 config와 template로만 만든다.
- 신규 결과물 추가 시 `코드 수정 최소, config/template 추가 최대` 원칙을 유지한다.

### 4.5 Streamlit 역할 정의
- Streamlit은 대시보드가 아니라 `작업 콘솔`이다.
- 역할: 파일 업로드, 시나리오 선택, 실행 트리거, 로그 확인, 결과 다운로드
- 무거운 집계/렌더는 Streamlit 내부에서 직접 처리하지 않고 공통 엔진 호출로 처리한다.
- 최종 산출물은 Streamlit 화면이 아니라 HTML 파일이다.

---

## 5) 표준 실행 흐름 (실무 운영)

1. 웹에서 파일 업로드
- 원천 파일 업로드(도메인 구분: 예 `sales`, `targets`, `crm`, `market`, `inventory` ...)

2. 시나리오 선택
- 예: `monthly`, `quarterly`, `business`

3. config 로드
- 선택 시나리오의 입력/키/지표/템플릿/출력 규칙 로드

4. 공통 엔진 실행
- 로딩 -> 표준화 -> 키검증 -> 조인 -> 지표계산 -> payload 생성

5. 템플릿 주입
- config가 지정한 템플릿에 payload 렌더

6. 산출물/로그 저장 및 배포
- 결과 파일 저장
- 검증 로그와 품질 메타데이터 저장
- 웹에서 HTML 다운로드 제공

핵심:
- 결과물이 10개여도 엔진은 1개다.
- 운영 단위는 "코드 10개"가 아니라 "시나리오 10개"다.

---

## 6) 데이터 표준화 규칙

### 6.1 컬럼 표준화 규칙
- 원천 컬럼 -> 표준 컬럼명으로 먼저 변환
- 표준 컬럼명은 영문 snake_case 권장
- 이름 컬럼(병원명/품목명/활동명)은 설명 속성으로 유지

### 6.2 ID 규칙 (강제)
- 조인/집계는 이름이 아닌 ID 기반 수행
- 필수 ID 예시:
  - `hospital_id`
  - `product_id`
  - `branch_id`
  - 필요 시 `activity_id`
- 시간키는 포맷 고정: `metric_month=YYYY-MM`

### 6.3 HIR 등 다중지표 규칙
- 권장 1: Long (`metric_code`, `metric_value`)
- 권장 2: 보고 직전에 Wide 피벗 (`HIR`, `RTR`, ...)
- 지표를 무리하게 각각 별도 테이블 조인하지 않기

---

## 7) 조인 설계 규칙 (가장 중요)

### 7.1 조인 전에 반드시 결정할 것
- 기준 마스터 grain: 예) `hospital_id + metric_month + product_id`
- 각 입력 테이블을 이 grain으로 변환 가능한지 확인

### 7.2 조인 전 검증
- 키 null 비율
- 키 중복 건수
- 예상 행 수 대비 증감률

### 7.3 조인 실패 패턴
- N:N 조인으로 행 폭증
- 이름 기반 조인으로 누락/중복
- 월 포맷 불일치(`2026-1` vs `2026-01`)

### 7.4 조인 게이트 (Fail-Fast)
아래 중 하나라도 위반하면 즉시 중단:
- 필수 키 null 비율 > 임계치
- 조인키 중복 발견
- 조인 후 행 수 증가율 임계 초과

---

## 8) 실행 계획 (도입 로드맵)

### Phase 1. 기준선 확정 (1~2주)
- 표준 컬럼 사전 확정
- 필수 ID 키 사전 확정
- 시나리오 3종(월간/분기/업무) config 초안

### Phase 2. 엔진 분리 (1~2주)
- `loaders/mapping/keys/metrics/payload/render` 모듈 분리
- 엔진 실행 단계별 산출물 저장 구조 도입
- 검증 게이트 구현

### Phase 3. 시나리오 확장 (1주)
- config 10개로 확장
- 템플릿 주입 구조 정리
- Streamlit 시나리오 선택/업로드 UI 연동

### Phase 4. 운영 안정화 (지속)
- 실행 로그 뷰어(작업 콘솔 내)
- 품질 리포트(누락률/중복률/실패율)
- 매핑 사전 관리 프로세스 도입

---

## 9) 권장 파일/모듈 구조

```text
config/
  mapping.json                    # 컬럼 별칭 사전
  domains.yaml                    # 사용 도메인 레지스트리
  scenarios/
    monthly.yaml
    quarterly.yaml
    business.yaml
  schema/
    master_columns.yaml
    key_policy.yaml
    domains/
      sales.yaml
      targets.yaml
      crm.yaml
      market.yaml                 # (예시) 신규 도메인
      inventory.yaml              # (예시) 신규 도메인

scripts/
  engine/
    orchestrator.py               # run_pipeline(config)
    loaders.py                    # 파일 탐색/로딩
    mapping.py                    # 매핑/표준화
    keys.py                       # 키 생성/검증/조인
    metrics.py                    # 지표 계산
    payload.py                    # 템플릿용 데이터 변환
    render.py                     # 템플릿 렌더/출력
  apps/
    streamlit_app.py              # 운영 UI (파일업로드/실행/다운로드)

templates/
  base/
    layout.html
    components/
      header.html
      chart_block.html
      table_block.html
  monthly_report.html
  quarterly_report.html
  business_report.html

output/
  processed/
  reports/
  logs/

data/
  {domain}/                       # 도메인별 원천 데이터 폴더 (동적 확장)
  logic/                          # 가중치/시스템 설정 등 로직 참조 데이터
  public/                         # 공용 참조 데이터(코드/룩업/매핑 보조)
```

구조 원칙:
- 핵심 비즈니스 로직은 `scripts/engine/*`에만 존재한다.
- 앱(UI)은 엔진을 호출만 하고 로직을 복제하지 않는다.
- 시나리오 확장은 `config/scenarios/*`와 `templates/*` 추가로 끝낸다.
- 도메인 확장은 `data/{domain}` + `config/schema/domains/{domain}.yaml` 추가로 끝낸다.

---

## 10) Config 설계 표준

### 10.1 시나리오 config 최소 항목
- `scenario_name`
- `domains`
- `input_sources` (domain별 경로 또는 패턴)
- `required_columns`
- `join_keys`
- `grain`
- `filters`
- `metrics`
- `template`
- `output_name_rule`

### 10.2 예시 스키마
```yaml
scenario_name: monthly
domains: [sales, targets, crm]
input_sources:
  sales: data/sales/*.{csv,xlsx}
  targets: data/targets/*.{csv,xlsx}
  crm: data/crm/*.{csv,xlsx}

grain: [hospital_id, metric_month, product_id]
join_keys:
  sales_targets: [hospital_id, metric_month, product_id]
  sales_crm: [hospital_id, metric_month]

filters:
  metric_month: current_month

metrics:
  - sales_amount
  - target_amount
  - hir
  - rtr

template: templates/monthly_report.html
output_name_rule: Strategic_Full_Dashboard_{date}
```

---

## 11) 운영 규칙

### 11.1 매핑 사전 운영
- 새 컬럼 별칭은 검토 후 등록
- 기존 표준키 삭제 금지(하위 호환 유지)
- 변경 이력 기록(누가/언제/왜)

### 11.2 데이터 보호
- `data/`, `output/` 원본 삭제/초기화 금지
- 결과물은 날짜+충돌회피 suffix 규칙 유지

### 11.3 인코딩 규칙
- UTF-8 고정
- 한글 깨짐 출력은 수정 근거로 사용 금지
- 한글 관련 검증 전 `PYTHONIOENCODING=utf-8` 설정

### 11.4 장애 대응
- 실패 단계와 원인 키를 로그에 남김
- 재실행 시 동일 config로 재현 가능해야 함

---

## 12) 품질 체크리스트 (실행 전/후)

### 실행 전
- 시나리오 config 존재 여부
- 도메인별 입력 파일 존재 여부
- 필수 컬럼 매핑 가능 여부

### 실행 중
- 키 null/중복 검증 통과
- 조인 후 행 수 이상치 여부
- 지표 계산 결측률

### 실행 후
- 산출물 파일 생성 여부
- 핵심 지표 범위 sanity check
- 로그/메타데이터 저장 여부

---

## 13) 자주 발생하는 실수와 예방책

1. 실수: 매핑만 하면 조인이 자동 안정화된다고 생각
- 예방: 조인키/그레인 규칙을 별도로 강제

2. 실수: 병원명/품목명으로 직접 조인
- 예방: ID 부여 후 ID로만 조인

3. 실수: Long 데이터를 바로 Wide 기준 테이블과 조인
- 예방: 먼저 동일 grain으로 집계/피벗

4. 실수: 시나리오마다 엔진 코드 복제
- 예방: 엔진 1개 + config 분기

5. 실수: 결과물만 확인하고 중간 검증 생략
- 예방: 매핑/조인/지표 계산 단계 산출물 필수 저장

---

## 14) KPI (자동화 성숙도 측정)

- 신규 시나리오 추가 리드타임
- 수동 수정 없는 성공 실행률
- 조인 오류 재발률
- 매핑 사전 재사용률
- 결과 생성 소요 시간

---

## 15) 최종 결론

이 프로젝트의 올바른 확장 방향은 다음 한 줄로 요약된다.

- **표준 컬럼으로 통일하고, ID로 조인하며, 시나리오 config와 템플릿으로 결과를 분기하는 공통 엔진 구조를 유지한다.**

이 원칙을 지키면 데이터/요구사항이 자주 바뀌는 실무에서도 안정적으로 자동화를 운영할 수 있다.
