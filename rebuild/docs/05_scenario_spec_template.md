# 05 Scenario Spec Template

## 목적
`monthly`, `quarterly`, `business` 시나리오 정의 템플릿을 표준화한다.

## 필수 항목
- `scenario_name`
- `domains`
- `input_sources`
- `grain`
- `join_keys`
- `filters`
- `metrics`
- `template`
- `output_name_rule`

## 작성 템플릿
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

## 규칙
- scenario별 파일 분리
- 공통 규칙은 별도 공통 config로 분리
- 하드코딩 금지
