# Map Join Grain Policy

## 목적
맵빌더 조인 품질을 보장하기 위해 기준 grain, 조인키, 임계치를 정의한다.

## 기준 Grain
- `map_master`: CRM 이벤트 row + 월 기준 실적/목표 결합
- `markers`: `rep + metric_month + activity_date + hospital`
- `routes`: `rep + metric_month + activity_date`

## 조인키 우선순위
1. `hospital_id + metric_month`
2. `hospital_name(normalized) + metric_month` (fallback)

## 사전 검증
- 키 null 비율
- 키 중복 건수
- 월 포맷 정합성(`YYYY-MM`)

## Fail-Fast 임계치(기본값)
- 키 null 비율 > 5%: 중단
- 조인키 중복 > 0: 중단
- 조인 후 row 증가율 > 120%: 중단
- 병원키 미매핑률 > 2%: 중단

## 로그 필수 항목
- 조인명
- 키 목록
- 전/후 row count
- 미매핑률
- 실패 사유

