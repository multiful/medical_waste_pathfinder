
# pages/2_🚚_운송최적화_CVRP.py
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

from vrp_utils import (
    load_cvrp_data,
    solve_multi_depot_cvrp,
    build_map_data,
)

# -------------------------------------------------
# 기본 설정
# -------------------------------------------------
st.set_page_config(
    page_title="의료폐기물 운송 최적화 (CVRP)",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚚 의료폐기물 운송 최적화 (CVRP) + 지도 시각화")

# -------------------------------------------------
# 1) 데이터 로더 (캐시)
# -------------------------------------------------
@st.cache_data
def load_data():
    # 프로젝트 구조에 맞춰 경로만 조정하면 됨
    data_dir = Path(".")  # 또는 Path("data")
    cvrp_path = data_dir / "cvrp_master_db.csv"
    nodes_path = data_dir / "all_nodes.csv"

    cvrp_df, nodes_df = load_cvrp_data(
        str(cvrp_path),
        str(nodes_path),
    )
    return cvrp_df, nodes_df


cvrp_df, nodes_df = load_data()

# -------------------------------------------------
# 2) 사이드바: 시나리오 선택
# -------------------------------------------------
st.sidebar.header("⚙️ 시뮬레이션 설정")

year = st.sidebar.selectbox(
    "연도 선택",
    options=sorted(cvrp_df["연도"].unique()),
    index=0,
)

month = st.sidebar.selectbox(
    "월 선택",
    options=sorted(cvrp_df["월"].unique()),
    index=0,
)

weekday = st.sidebar.selectbox(
    "요일 선택",
    options=list(cvrp_df["요일"].unique()),
    index=0,
)

vehicle_capacity = st.sidebar.number_input(
    "차량 적재 용량 (kg)",
    min_value=1_000,
    max_value=20_000,
    step=1_000,
    value=8_000,
)

vehicles_per_depot = st.sidebar.slider(
    "각 소각장(Depot)별 차량 수",
    min_value=1,
    max_value=10,
    value=3,
)

run_button = st.sidebar.button("🚀 최적 경로 계산 실행")

# -------------------------------------------------
# 3) 실행 + 결과 표시
# -------------------------------------------------
if run_button:
    with st.spinner("CVRP 최적 경로 계산 중..."):
        try:
            all_routes, routes_per_customer, summary = solve_multi_depot_cvrp(
                cvrp_df=cvrp_df,
                nodes_df=nodes_df,
                year=year,
                month=month,
                weekday=weekday,
                vehicle_capacity=vehicle_capacity,
                vehicles_per_depot=vehicles_per_depot,
            )
        except Exception as e:
            st.error(f"CVRP 최적화 중 오류가 발생했습니다:\n\n{e}")
            st.stop()

    st.success("최적 경로 계산 완료 ✅")

    # 3-1. 시도/시군구별 리스크 요약 테이블
    st.subheader("📊 시도·시군구별 운송 리스크 요약")
    if not summary.empty:
        st.dataframe(
            summary.style.format(
                {
                    "total_demand": "{:,.0f}",
                    "served_demand": "{:,.0f}",
                    "unserved_demand": "{:,.0f}",
                    "served_ratio": "{:.2%}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info("요약 데이터가 없습니다.")

    # 3-2. 지도 시각화용 데이터 생성
    depots_df, customers_df, lines_df = build_map_data(
        nodes_df=nodes_df,
        routes_per_customer=routes_per_customer,
        all_routes=all_routes,
    )

    st.subheader("🗺️ 노드/경로 지도 시각화")

    if customers_df.empty:
        st.info("표시할 고객 노드가 없습니다.")
    else:
        # 초기 화면 중심 (대한민국 중심 근처)
        view_state = pdk.ViewState(
            latitude=float(customers_df["Lat"].mean()),
            longitude=float(customers_df["Lng"].mean()),
            zoom=6,
            pitch=0,
        )

        # 고객 노드 레이어 (서비스 여부에 따라 색상)
        node_layer = pdk.Layer(
            "ScatterplotLayer",
            data=customers_df,
            get_position="[Lng, Lat]",
            get_radius="size",
            pickable=True,
            get_fill_color="""
            [kind == 'Unserved' ? 200 : 0,
             kind == 'Served' ? 120 : 0,
             150, 180]
            """,
        )

        # 디포트 레이어 (검은색)
        depot_layer = pdk.Layer(
            "ScatterplotLayer",
            data=depots_df,
            get_position="[Lng, Lat]",
            get_radius="size",
            pickable=True,
            get_fill_color="[0, 0, 0, 220]",
        )

        # 차량 경로 라인 레이어
        line_layer = pdk.Layer(
            "LineLayer",
            data=lines_df,
            get_source_position="[start_lng, start_lat]",
            get_target_position="[end_lng, end_lat]",
            get_width=3,
            get_color="[50, 50, 50, 180]",
            pickable=False,
        )

        tooltip = {
            "html": "<b>{Name}</b><br/>"
                    "종류: {kind}<br/>"
                    "일일 폐기물: {Daily_Demand_Kg} kg",
            "style": {"backgroundColor": "white", "color": "black"},
        }

        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",  # 토큰 없으면 기본 스타일로 표시
            initial_view_state=view_state,
            layers=[line_layer, depot_layer, node_layer],
            tooltip=tooltip,
        )

        st.pydeck_chart(deck, use_container_width=True)

    # 3-3. 원시 경로 정보 (디버그/설명용)
    with st.expander("🔍 상세 경로 정보 (디버그용)"):
        st.write("차량별 경로 리스트 (depot → 고객들 → depot)")
        st.json(all_routes)

else:
    st.info("왼쪽 사이드바에서 조건을 설정하고 **'🚀 최적 경로 계산 실행'** 버튼을 눌러주세요.")
