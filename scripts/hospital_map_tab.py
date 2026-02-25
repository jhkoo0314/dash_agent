import os
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.components.v1 import declare_component


def render_hospital_map_tab(df: pd.DataFrame, current_dir: str, map_path: str | None = None) -> None:
    """Render the hospital map tab UI using the existing base map HTML."""
    if map_path is None:
        map_path = os.path.join("c:\\", "agent_b", "hospital_map.html")
    st.markdown("#### 🗺️ 전국병원 지도 뷰")
    st.info("기존 병원 마커 위에 현재 실적 데이터가 오버레이 됩니다. (파이썬에서 지도를 매번 연산하지 않고, 프론트엔드 단에서 DOM을 조작하여 데이터를 덮어씌웁니다.)")
    
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            html_data = f.read()
            
        # --- [데이터 오버레이 로직] ---
        if '병원명' in df.columns:
            hosp_df = df.groupby('병원명').agg({
                '처방금액': 'sum',
                '처방수량': 'sum',
                '성명': lambda x: ', '.join(x.dropna().unique())
            }).reset_index()
            
            import json
            overlay_data = {}
            for _, row in hosp_df.iterrows():
                h_key = str(row['병원명']).strip()
                if h_key and h_key != 'nan':
                    overlay_data[h_key] = {
                        "처방금액": int(row['처방금액']) if pd.notnull(row['처방금액']) else 0,
                        "처방수량": int(row['처방수량']) if pd.notnull(row['처방수량']) else 0,
                        "담당자": str(row['성명']).strip()
                    }
            
            overlay_json = json.dumps(overlay_data, ensure_ascii=False)
            
            # HTML 템플릿에 스크립트 삽입
            inject_script = f"""
            <script>
            (function() {{
                const overlayData = {overlay_json};
                const overlayKeys = Object.keys(overlayData);
                let matchCount = 0;
                let markersFound = 0;
                let attempts = 0;

                function startOverlay() {{
                    attempts++;
                    let allMarkers = [];
                    
                    // 1. 모든 전역 객체 뒤져서 마커/클러스터/맵 찾기
                    for (let key in window) {{
                        let obj = window[key];
                        if (!obj) continue;
                        
                        // 맵 또는 클러스터 그룹인 경우
                        if (obj.eachLayer && (key.startsWith('map_') || key.startsWith('marker_cluster_'))) {{
                            obj.eachLayer(layer => {{
                                if (layer.getTooltip) allMarkers.push(layer);
                                if (layer.eachLayer) {{ // 클러스터 내부 재탐색
                                    try {{
                                        layer.eachLayer(sub => {{ if(sub.getTooltip) allMarkers.push(sub); }});
                                    }} catch(e) {{}}
                                }}
                            }});
                        }}
                        
                        // 개별 마커인 경우 (marker_...)
                        if (key.startsWith('marker_') && obj.getTooltip) {{
                            allMarkers.push(obj);
                        }}
                    }}
                    
                    // 중복 제거
                    allMarkers = [...new Set(allMarkers)];
                    markersFound = allMarkers.length;

                    if (markersFound === 0 && attempts < 10) {{
                        // 아직 지도가 안 그려졌으면 1초 뒤 재시도
                        setTimeout(startOverlay, 1000);
                        updateStatus("⏳ 지도를 로드 중입니다... (" + attempts + "/10)");
                        return;
                    }}

                    allMarkers.forEach(marker => {{
                        const tt = marker.getTooltip();
                        if (!tt) return;
                        
                        const content = tt.getContent();
                        const div = document.createElement('div');
                        div.innerHTML = content;
                        const hospName = div.innerText.trim();
                        const hospNorm = hospName.replace(/\\s+/g, '').toLowerCase();
                        
                        let matchedKey = null;
                        for(let i=0; i<overlayKeys.length; i++) {{
                            let keyNorm = overlayKeys[i].replace(/\\s+/g, '').toLowerCase();
                            if(hospNorm.indexOf(keyNorm) !== -1 || keyNorm.indexOf(hospNorm) !== -1) {{
                                matchedKey = overlayKeys[i];
                                break;
                            }}
                        }}
                        
                        if (matchedKey) {{
                            matchCount++;
                            const d = overlayData[matchedKey];
                            
                            // 🌟 Streamlit 연동: 마커 클릭 시 병원명 전송
                            marker.on('click', function() {{
                                if (typeof window.setComponentValue === 'function') {{
                                    window.setComponentValue(matchedKey);
                                }}
                            }});

                            const pop = marker.getPopup();
                            if (pop) {{
                                const appendHtml = "<hr><h5 style='color:#0d6efd; font-weight:bold; margin-top:10px;'>🔹 실적 달성 현황</h5>" +
                                                 "<div style='font-size:13px;'>" +
                                                 "<b>실적금액:</b> <span style='color:red;'>" + d.처방금액.toLocaleString() + " 원</span><br>" +
                                                 "<b>담당자:</b> " + d.담당자 + "</div>";
                                
                                const currentContent = pop.getContent();
                                if (typeof currentContent === 'string' && currentContent.indexOf('실적 달성 현황') === -1) {{
                                    pop.setContent(currentContent.replace('</div>', appendHtml + '</div>'));
                                }} else if (currentContent instanceof HTMLElement && currentContent.innerHTML.indexOf('실적 달성 현황') === -1) {{
                                    currentContent.innerHTML += appendHtml;
                                }}
                                
                                if (window.L && L.AwesomeMarkers) {{
                                    marker.setIcon(L.AwesomeMarkers.icon({{
                                        markerColor: 'green', iconColor: 'white', icon: 'star', prefix: 'fa'
                                    }}));
                                }}
                            }}
                        }}
                    }});

                    updateStatus("<b>✅ 오버레이 완료</b><br>찾은 마커: " + markersFound + "<br>매칭 성공: " + matchCount);
                }}

                function updateStatus(msg) {{
                    let d = document.getElementById('debug-box');
                    if (!d) {{
                        d = document.createElement('div');
                        d.id = 'debug-box';
                        d.style.cssText = "position:absolute;top:10px;left:50px;z-index:9999;background:white;padding:12px;border:2px solid #0d6efd;border-radius:10px;font-family:sans-serif;box-shadow:0 4px 10px rgba(0,0,0,0.2);min-width:150px;";
                        document.body.appendChild(d);
                    }}
                    d.innerHTML = "<b>🔍 분석 엔진 가동</b><br>" + 
                                 "<span style='font-size:12px;'>데이터 병원: " + overlayKeys.length + "개</span><br>" + msg;
                }}

                // 초기 실행
                setTimeout(startOverlay, 2000);
                
                // --- Streamlit Component Bridge ---
                function sendMsg(type, data) {{
                    var outData = Object.assign({{isStreamlitMessage: true, type: type}}, data);
                    window.parent.postMessage(outData, "*");
                }}
                sendMsg("streamlit:componentReady", {{apiVersion: 1}});
                sendMsg("streamlit:setFrameHeight", {{height: 750}});
                window.setComponentValue = function(val) {{
                    sendMsg("streamlit:setComponentValue", {{value: val}});
                }};
            }})();
            </script>
            """
            html_data += inject_script
        else:
            st.warning("⚠️ '병원명' 매핑이 안 되었습니다. STEP 1에서 요양기관명을 '병원명'으로 선택해주세요.")
        # ------------------------------
        
        # --- [Component 렌더링 및 딥다이브 연동] ---
        map_dir = os.path.join(current_dir, "map_component")
        os.makedirs(map_dir, exist_ok=True)
        with open(os.path.join(map_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(html_data)
            
        from streamlit.components.v1 import declare_component
        map_component = declare_component("hospital_map", path=map_dir)
        
        c1, c2 = st.columns([7, 3])
        
        with c1:
            st.markdown("##### 📍 전국 병원 분포 (마커를 클릭하세요)")
            clicked_hosp = map_component(key="hosp_map_comp")
            
        with c2:
            st.markdown("### 🔍 딥다이브 플래시보드")
            if clicked_hosp:
                st.success(f"**{clicked_hosp}** 상세 분석")
                target_df = df[df['병원명'].astype(str).str.strip() == clicked_hosp]
                
                if not target_df.empty:
                    # 1. 핵심 KPI
                    st.metric("총 처방금액", f"{int(target_df['처방금액'].sum()):,} 원")
                    rep_name = ', '.join(target_df['성명'].dropna().unique())
                    st.caption(f"**담당자:** {rep_name}")
                    st.divider()
                    
                    # 2. 벤치마크 분석 (지점 평균 대비)
                    st.markdown("##### 📊 지표 밸런스 (지점 평균 대비 %)")
                    st.caption("소속 지점 평균을 100%로 잡았을 때의 상대적 수준입니다.")
                    
                    target_branch = target_df['지점'].iloc[0] if '지점' in target_df.columns else None
                    avg_kpi = target_df[['HIR_Raw', 'RTR_Raw', 'PHR_Raw', 'PI_Raw']].mean().fillna(0)
                    
                    if target_branch:
                        branch_avg = df[df['지점'] == target_branch][['HIR_Raw', 'RTR_Raw', 'PHR_Raw', 'PI_Raw']].mean().replace(0, 1)
                        bench_values = [
                            (avg_kpi['HIR_Raw'] / branch_avg['HIR_Raw']) * 100,
                            (avg_kpi['RTR_Raw'] / branch_avg['RTR_Raw']) * 100,
                            (avg_kpi['PHR_Raw'] / branch_avg['PHR_Raw']) * 100,
                            (avg_kpi['PI_Raw'] / branch_avg['PI_Raw']) * 100
                        ]
                    else:
                        bench_values = [100, 100, 100, 100]
                        
                    categories = ['활동질(HIR)', '관계온도(RTR)', '파이프라인(PHR)', '성과지수(PI)']
                    
                    import plotly.express as px
                    fig = px.bar(
                        x=bench_values,
                        y=categories,
                        orientation='h',
                        text=[f"{v:.1f}%" for v in bench_values],
                        color=bench_values,
                        color_continuous_scale='RdYlGn', # 낮은값 빨강, 높은값 초록
                        range_x=[0, max(200, max(bench_values)*1.1)]
                    )
                    fig.add_vline(x=100, line_dash="dash", line_color="gray", annotation_text="지점 평균")
                    fig.update_layout(
                        xaxis_title="", yaxis_title="",
                        height=250, margin=dict(l=20, r=20, t=20, b=20),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.divider()

                    # 3. [신규] 활동-실적 인사이트 진단
                    st.markdown("##### 💡 활동-실적 전략 인사이트")
                    
                    # 효율성 계산: 총 실적 / 활동 횟수
                    total_act = len(target_df.dropna(subset=['activities']))
                    total_sales = target_df['처방금액'].sum()
                    efficiency = total_sales / (total_act if total_act > 0 else 1)
                    
                    # 지점 평균 효율성
                    branch_total_act = len(df[df['지점'] == target_branch].dropna(subset=['activities']))
                    branch_total_sales = df[df['지점'] == target_branch]['처방금액'].sum()
                    branch_efficiency = branch_total_sales / (branch_total_act if branch_total_act > 0 else 1)
                    
                    insight_cols = st.columns(2)
                    with insight_cols[0]:
                        st.metric("활동 효율성", f"{efficiency / 1000:,.1f}k", 
                                  delta=f"{(efficiency/branch_efficiency-1)*100:.1f}%" if branch_efficiency>0 else None,
                                  help="방문 1회당 발생하는 처방금액 효율 (지점 평균 대비)")
                    
                    # 텍스트 오토-가이드 생성 (Rule-based)
                    advice = ""
                    if (efficiency > branch_efficiency) and (avg_kpi['PI_Raw'] > 100):
                        advice = "✅ **고효율 관리 모델:** 적은 방문으로도 높은 성과를 내고 있습니다. 현재의 활동 질(HIR)을 유지하며 리소스 여력을 경쟁 병원 침투에 활용하십시오."
                    elif (efficiency < branch_efficiency) and (avg_kpi['HIR_Raw'] > branch_avg['HIR_Raw']):
                        advice = "⚠️ **활동 과잉 신호:** 활동의 질은 좋으나 실적 전환율이 낮습니다. 단순 방문보다는 처방 의사결정권자와의 관계 심화(RTR)가 필요합니다."
                    elif (avg_kpi['PHR_Raw'] < branch_avg['PHR_Raw']):
                        advice = "🚨 **파이프라인 경고:** 미래 성과 지표(PHR)가 낮습니다. 현재 실적 유지에만 급급할 수 있으니, 신규 품목 제안 활동을 즉시 강화하십시오."
                    else:
                        advice = "📈 **안정적 성장세:** 지점 평균 수준의 밸런스를 유지하고 있습니다. 주기적인 방문 간격을 유지하며 이탈 방지에 주력하십시오."
                    
                    st.info(advice)
                    st.divider()
                    
                    # 4. 최근 활동 타임라인 (하단 이동)
                    st.markdown("##### 📅 최근 리얼 활동 로그")
                    act_df = target_df.dropna(subset=['activities']).sort_values('날짜', ascending=False).head(3)
                    if not act_df.empty:
                        for _, act_row in act_df.iterrows():
                            dt_str = act_row['날짜'].strftime('%m/%d') if pd.notnull(act_row['날짜']) else 'N/A'
                            st.markdown(f"📌 `{dt_str}` | {act_row['activities']}")
                    else:
                        st.caption("기록된 활동이 없습니다.")
                else:
                    st.warning("선택된 병원의 상세 데이터가 존재하지 않습니다.")
            else:
                st.info("☝️ 왼쪽 지도에서 반짝이는 초록색 마커를 클릭하시면 이곳에 담당자의 상세 분석 결과가 나타납니다.")
    else:
        st.warning(f"설정된 맵 파일({map_path})을 아직 찾을 수 없습니다. (먼저 지도를 생성해주세요)")
