# Map Test Plan

## 목적
맵빌더의 단위/통합/회귀 테스트 범위와 합격 기준을 정의한다.

## 단위 테스트
- 키 정규화 함수
- 월 파싱 함수
- 거리 계산 함수(haversine)
- payload 생성 함수(markers/routes)
- placeholder 치환 함수

## 통합 테스트
- 입력 4도메인 -> `map_master` 생성
- `map_master` -> `Spatial_Preview` 생성
- 좌표 결측률 임계치 초과 시 Fail-Fast
- placeholder 누락 템플릿에서 Render 실패 처리

## 회귀 테스트
- 고정 샘플셋 기준:
  - markers count
  - routes count
  - avg_route_km
- 허용 편차:
  - count ±5%
  - avg_route_km ±10%

## 테스트 데이터셋
- 정상 케이스 1세트
- 좌표 결측 케이스 1세트
- 조인키 충돌 케이스 1세트
- 월 포맷 불일치 케이스 1세트

## 합격 기준
- 치명 오류 0건
- Fail-Fast 동작 검증 완료
- 회귀 편차 허용 범위 이내

