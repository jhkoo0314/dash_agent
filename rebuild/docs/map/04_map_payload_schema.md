# Map Payload Schema

## 목적
맵 마스터 CSV와 지도 주입 JSON(`markers`, `routes`) 스키마를 고정한다.

## map_master CSV 스키마
- `activity_date` (string, `YYYY-MM-DD`)
- `metric_month` (string, `YYYY-MM`)
- `rep_name` (string)
- `hospital_name` (string)
- `target_amount` (number)
- `sales_amount` (number)
- `lon` (number)
- `lat` (number)
- `source_row_no` (int)

## markers JSON 스키마
```json
{
  "hospital": "string",
  "rep": "string",
  "month": "YYYY-MM",
  "date": "YYYY-MM-DD",
  "lat": 0.0,
  "lon": 0.0,
  "sales": 0.0,
  "target": 0.0,
  "seq": 1
}
```

## routes JSON 스키마
```json
{
  "rep": "string",
  "month": "YYYY-MM",
  "date": "YYYY-MM-DD",
  "coords": [
    { "seq": 1, "hospital": "string", "lat": 0.0, "lon": 0.0 }
  ],
  "total_km": 0.0,
  "visits": 1
}
```

## 스키마 검증 규칙
- 필수 필드 누락 금지
- `lat [-90, 90]`, `lon [-180, 180]`
- `seq`는 양의 정수
- `visits == len(coords)`

