# Map Quality Checklist

## 목적
맵빌더 실행 전/중/후 품질 점검 항목을 정의한다.

## 실행 전
- [ ] 시나리오 config 존재
- [ ] 4개 입력 도메인 파일 존재
- [ ] 필수 컬럼 매핑 가능
- [ ] `metric_month` 포맷 점검

## 실행 중
- [ ] 키 null 비율 임계치 이내
- [ ] 조인키 중복 없음
- [ ] 조인 후 row 증가율 이상치 없음
- [ ] 좌표 결측률 임계치 이내
- [ ] 좌표 범위 이탈률 임계치 이내

## 실행 후
- [ ] `map_master` 생성 확인
- [ ] `Spatial_Preview` 생성 확인
- [ ] markers/routes 개수 sanity check
- [ ] run_id 로그 저장 확인
- [ ] 동일 조건 재실행 재현성 확인

## 회귀 기준(권장)
- markers 총개수 편차: 기준 대비 ±5%
- routes 총개수 편차: 기준 대비 ±5%
- 평균 거리(`avg_route_km`) 편차: 기준 대비 ±10%

