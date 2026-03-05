# Map Scenario Spec

## 목적
`map_spatial_preview` 시나리오 구성 규칙을 정의한다.

## YAML 예시
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
- quality gate는 config로 관리(하드코딩 금지)
- scenario 변경 시 영향분석/롤백 계획 동반

