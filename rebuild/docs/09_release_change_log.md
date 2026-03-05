# 09 Release Change Log

## 목적
config/템플릿/엔진 변경 이력, 영향 범위, 롤백 기준을 관리한다.

## 변경 유형
- Config 변경
- Template 변경
- Engine 코드 변경

## 기록 템플릿
```text
[Change ID]
- Date:
- Author:
- Type: config | template | engine
- Summary:
- Affected Scenarios:
- Affected Domains:
- Validation Result:
- Rollback Plan:
```

## 영향 분석 기준
- 시나리오별 결과 지표 변화
- 렌더 구조 변화
- 하위호환 영향(`mapping.json`)

## 롤백 원칙
- 즉시 복구 가능한 이전 버전 참조점 유지
- 데이터 원본/기존 산출물은 삭제하지 않음
- 실패 이슈와 조치 결과를 로그에 연계
