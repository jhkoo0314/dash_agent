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

## Map 시나리오 예시
```yaml
scenario_name: map_spatial_preview
domains: [crm, sales, targets, logic]
input_sources:
  crm: data/crm/*.{csv,xlsx}
  sales: data/sales/*.{csv,xlsx}
  targets: data/targets/*.{csv,xlsx}
  logic: data/logic/*.{csv,xlsx}

grain:
  map_master: [hospital_id, metric_month, activity_date, rep_id]
  route: [rep_id, metric_month, activity_date]

join_keys:
  crm_sales_targets: [hospital_id, metric_month]
  with_coords: [hospital_id]

filters:
  metric_month: current_month
  require_valid_coords: true

metrics:
  - visits
  - total_km
  - sales_amount
  - target_amount

template: templates/spatial_preview_template.html
output_name_rule:
  map_master_csv: map_master_{date}
  spatial_preview_html: Spatial_Preview_{date}_{time}

quality_gates:
  activity_date_parse_fail_rate_max: 0.05
  coord_missing_rate_max: 0.10
  coord_out_of_range_rate_max: 0.01
  join_row_growth_rate_max: 1.20
  hospital_unmatched_rate_max: 0.02
```

## 규칙
- scenario별 파일 분리
- 공통 규칙은 별도 공통 config로 분리
- 하드코딩 금지
