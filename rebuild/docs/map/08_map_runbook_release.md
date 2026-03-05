# Map Runbook and Release

## 목적
맵빌더 운영 절차, 장애 대응, 릴리즈/롤백 기준을 정의한다.

## 일상 운영 절차
1. 입력 반입 확인
2. `map_spatial_preview` 시나리오 선택
3. 1단계 `map_master` 생성
4. 품질 게이트 결과 확인
5. 2단계 `Spatial_Preview` 생성
6. 결과 검토 후 배포

## 장애 대응
1. 실패 stage 식별
2. 원인 분류(입력/매핑/조인/좌표/렌더)
3. 수정 후 동일 조건 재실행
4. 장애 리포트 기록(run_id 연계)

## 변경관리 템플릿
```text
[Change ID]
- Date:
- Author:
- Type: config | template | engine | map
- Summary:
- Affected Scenarios:
- Validation Result:
- Rollback Plan:
```

## 롤백 원칙
- 즉시 복구 가능한 이전 기준점 유지
- 원본 데이터/기존 산출물 삭제 금지
- 실패 로그와 조치 이력 연결

