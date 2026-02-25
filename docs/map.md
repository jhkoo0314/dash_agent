# 전국병원 지도 최종 생성 구현 계획

## 1) 최종 목표
- 기준 지도는 루트의 `hospital_map.html`을 사용한다.
- `hospital_map.html`은 이미 `종합병원/상급종합` 마커가 포함된 베이스 지도 데이터다.
- 샌드박스에서는 이 베이스 지도 위에 실적/활동 오버레이를 쌓는다.
- 버튼 클릭 시 오버레이가 반영된 `최종 지도 HTML`을 생성한다.

## 2) 범위 재정의
### 포함
- `hospital_map.html` 로드
- 오버레이 데이터 생성(병원별 집계)
- 오버레이 스크립트 주입
- 최종 지도 파일 저장
- 샌드박스 버튼 연동

### 제외
- 베이스 병원 지도(`hospital_map.html`) 재생성 로직 고도화
- 병원 마스터 원천 데이터 구조 변경
- 좌표/지오코딩 신규 파이프라인 구축

## 3) 현재 구조 요약
- 대시보드: `scripts/sfe_sandbox.py`
- 리포트 생성 패턴: `scripts/report_builder_v12.py`의 `build_final_reports(...)`
- 지도 베이스 파일: `hospital_map.html`
- 지도 컴포넌트 출력 경로: `scripts/map_component/index.html`

핵심 문제:
- 현재는 지도 탭 안에 표시/오버레이/컴포넌트 코드가 섞여 있어 재사용성과 생성 흐름 관리가 어렵다.

## 4) 목표 아키텍처
## 4.1 신규 파일
1. `scripts/hospital_overlay_builder.py`
- 역할: 베이스 지도 + 오버레이 데이터로 최종 지도 HTML 생성
- 공개 함수: `build_final_hospital_map(df, external_config=None) -> str | None`

2. `scripts/hospital_map_tab.py`
- 역할: 지도 탭 렌더링(UI 전담)
- 공개 함수: `render_hospital_map_tab(df, map_path, current_dir) -> None`

## 4.2 기존 파일 변경
1. `scripts/sfe_sandbox.py`
- 지도 생성 버튼 추가
- 버튼 클릭 시 `build_final_hospital_map(...)` 호출
- 생성된 경로를 `render_hospital_map_tab(...)`에 전달

## 5) 생성 로직 상세
## 5.1 입력
- 베이스 지도: `hospital_map.html` (고정)
- 오버레이 원본: `st.session_state.clean_master` (`df`)

## 5.2 병원 오버레이 집계 규칙
- 키: `병원명` (trim 처리)
- 집계:
  - `처방금액`: sum
  - `처방수량`: sum
  - `성명`: unique join

예시 구조:
```json
{
  "서울성모병원": {
    "처방금액": 123456789,
    "처방수량": 321,
    "담당자": "홍길동, 김영희"
  }
}
```

## 5.3 HTML 주입 방식
1. `hospital_map.html` 읽기
2. 오버레이 JSON 직렬화(`ensure_ascii=False`)
3. JS 주입:
  - 마커 tooltip 이름과 오버레이 키 매칭
  - 매칭 시 팝업 콘텐츠 append
  - 마커 스타일 변경(예: green star)
  - 클릭 병원명 Streamlit으로 전달
4. 최종 HTML 저장

## 5.4 출력 정책
- 기본 출력: `output/Final_Hospital_Map_YYMMDD(.n).html`
- 필요 시 호환용 복사(선택):
  - `scripts/map_component/index.html`에 렌더용 동기화

## 6) 샌드박스 버튼 연동
## 6.1 UX
- 버튼 라벨: `🗺️ 최종 병원 지도(HTML) 생성`
- 스피너: `최종 지도 생성 중입니다...`
- 성공:
  - 생성 경로 출력
  - 다운로드 버튼 표시
- 실패:
  - `st.error(...)`로 사유 출력

## 6.2 실행 순서
1. `clean_master` 존재 확인
2. `build_final_hospital_map(df, external_config=CONFIG)` 호출
3. 반환된 `map_path`를 세션에 저장(`st.session_state.last_final_map`)
4. 지도 탭에서 `map_path` 우선 렌더

## 7) 유효성 검증
1. `병원명` 컬럼 존재 여부
2. `처방금액`, `처방수량`, `성명` 컬럼 존재 여부(없으면 안전 대체)
3. 베이스 파일 `hospital_map.html` 존재 여부
4. 집계 결과 0건일 때 경고 후 생성 중단

## 8) 단계별 작업 계획
1. `scripts/hospital_overlay_builder.py` 작성
2. `scripts/hospital_map_tab.py`로 탭 렌더 코드 이동
3. `scripts/sfe_sandbox.py`에 버튼-생성-다운로드 흐름 연결
4. 최종 지도 파일명 규칙 적용
5. 회귀 점검(리포트 생성 버튼 영향 없음 확인)

## 9) 테스트 체크리스트
1. `streamlit run scripts/sfe_sandbox.py` 실행
2. `최종 병원 지도(HTML) 생성` 버튼 클릭
3. `output/Final_Hospital_Map_YYMMDD(.n).html` 생성 확인
4. 지도 렌더링, 마커 오버레이, 클릭 연동 확인
5. 기존 리포트 생성(`build_final_reports`) 정상 동작 확인

## 10) Definition of Done
- 기준 지도는 `hospital_map.html`로 고정되어 사용된다.
- 샌드박스 버튼으로 오버레이 반영 최종 지도 HTML이 생성된다.
- 생성 산출물은 `output/`에 날짜 규칙으로 누적 저장된다.
- 기존 리포트/매핑/출력 흐름에 회귀가 없다.
