# 04 Join Grain Policy

## 목적
마스터 grain, 조인키, Fail-Fast 임계치를 표준화해 조인 품질을 보장한다.

## 마스터 Grain 정의
- 기준 grain은 시나리오마다 명시한다.
- 예: `hospital_id + metric_month + product_id`

## 조인키 규칙
- 이름 조인 금지, ID 조인 강제
- 시간키 포맷 통일 후 조인
- 조인 전 각 테이블을 기준 grain으로 정규화

## 사전 검증
- 키 null 비율
- 키 중복 건수
- 예상 행 수 대비 변동률

## Fail-Fast 임계치
- 필수 키 null 비율 > 임계치: 중단
- 조인키 중복 발견: 중단
- 조인 후 행 증가율 > 임계치: 중단

## 흔한 실패 패턴
- N:N 조인으로 행 폭증
- 월 포맷 불일치 (`2026-1`, `2026-01`)
- Long/Wide grain 불일치

## 로그 필수 항목
- 조인명
- 사용 키
- 전/후 row count
- 실패 사유
