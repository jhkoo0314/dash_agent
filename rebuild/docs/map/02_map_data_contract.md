# Map Data Contract

## 목적
맵빌더 입력/출력 데이터의 표준 컬럼, 타입, 포맷, 위반 처리 기준을 정의한다.

## 입력 도메인
- CRM: 활동 이벤트
- SALES: 월 실적
- TARGET: 월 목표
- COORD(LOGIC): 병원 좌표

## 필수 표준 컬럼
- `activity_date` (date)
- `metric_month` (`YYYY-MM`)
- `hospital_id` or `hospital_name`
- `rep_id` or `rep_name`
- `sales_amount` (numeric)
- `target_amount` (numeric)
- `lat`, `lon` (numeric)

## 타입/포맷
- ID: string
- 금액/수치: numeric
- 날짜: ISO date
- 월: `YYYY-MM`
- 텍스트/한글: UTF-8

## 결측/중복 정책
- 필수 키 결측률 임계치 초과 시 실행 중단
- 조인키 중복은 기본 불허(예외는 정책 문서 명시)
- 좌표 결측은 임계치 이하만 허용

## 계약 위반 처리
- Fail-Fast 중단
- 위반 컬럼/건수/샘플키를 로그에 기록

