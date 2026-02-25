import os
import pandas as pd
import streamlit as st

from map_data_builder import build_map_master_csv, build_spatial_preview_html_from_csv


def render_hospital_map_tab(df: pd.DataFrame, current_dir: str, map_path: str | None = None) -> None:
    st.markdown("#### 🗺️ 전국병원 지도 뷰")
    st.info(
        "지도는 계산 시간이 길어 미리보기를 생략합니다. "
        "먼저 맵데이터 CSV를 생성/검토한 뒤, 최종 HTML 생성을 실행하세요."
    )

    if "map_master_csv_path" not in st.session_state:
        st.session_state.map_master_csv_path = None
    if "map_html_path" not in st.session_state:
        st.session_state.map_html_path = None

    st.markdown("##### 1단계. 맵데이터 빌더 (CSV 생성)")
    if st.button("🧱 1단계 맵데이터 빌더 실행", type="primary"):
        with st.spinner("맵데이터를 병합/매핑하여 CSV를 생성 중입니다..."):
            try:
                csv_path, _ = build_map_master_csv()
                st.session_state.map_master_csv_path = csv_path
                st.success(f"✅ 맵데이터 CSV 생성 완료: `{csv_path}`")
            except Exception as e:
                st.error(f"❌ 맵데이터 CSV 생성 실패: {e}")

    csv_path = st.session_state.map_master_csv_path
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            st.download_button(
                label="📥 생성된 맵데이터 CSV 다운로드",
                data=f,
                file_name=os.path.basename(csv_path),
                mime="text/csv",
            )

        try:
            preview_df = pd.read_csv(csv_path, encoding="utf-8-sig")
            st.caption(f"미리보기 행 수: {len(preview_df):,} / 표시: 100행")
            st.dataframe(preview_df.head(100), use_container_width=True)
        except Exception as e:
            st.warning(f"CSV 미리보기 로드 실패: {e}")
    else:
        st.caption("맵데이터 CSV가 아직 생성되지 않았습니다.")

    st.divider()
    st.markdown("##### 2단계. 최종 지도 HTML 생성")
    if st.button("🗺️ 2단계 최종 HTML 생성"):
        if not csv_path or not os.path.exists(csv_path):
            st.warning("먼저 1단계 맵데이터 CSV를 생성해 주세요.")
        else:
            with st.spinner("최종 지도 HTML 생성 중입니다..."):
                try:
                    html_path = build_spatial_preview_html_from_csv(csv_path)
                    st.session_state.map_html_path = html_path
                    st.success(f"✅ 최종 지도 HTML 생성 완료: `{html_path}`")
                except Exception as e:
                    st.error(f"❌ 최종 지도 HTML 생성 실패: {e}")

    html_path = st.session_state.map_html_path
    if html_path and os.path.exists(html_path):
        with open(html_path, "rb") as f:
            st.download_button(
                label="📥 생성된 지도 HTML 다운로드",
                data=f,
                file_name=os.path.basename(html_path),
                mime="text/html",
            )
