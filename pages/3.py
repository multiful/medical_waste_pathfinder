# pages/vrp_utils.py
# -*- coding: utf-8 -*-
"""
의료폐기물 수요 데이터 뷰어 + 지도 시각화 페이지
- OR-Tools / vrp_utils 모듈 전혀 사용 안 함
- cvrp_master_db.csv + all_nodes.csv만 이용
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# -------------------------------------------------
# 기본 설정
# -------------------------------------------------
st.set_page_config(
    page_title="의료폐기물 수요 요약 (연도 × 월 × 요일)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 의료폐기물 수요 요약 (연도 × 월 × 요일) — 지도 시각화")

# -------------------------------------------------
# 1) 데이터 로더 (캐시)
# -------------------------------------------------
@st.cache_data
def load_data():
    base_dir = Path(".")  # 필요하면 Path("data") 등으로 변경

    cvrp_path = base_dir / "cvrp_master_db.csv"
    nodes_path = base_dir / "all_nodes.csv"

    df = pd.read_csv(cvrp_path)

    # Daily_Demand_Kg 컬럼 없고 Daily_Demand만 있으면 자동 변환
    if "Daily_Demand_Kg" not in df.columns:
        if "Daily_Demand" in df.columns:
            df["Daily_Demand_Kg"] = df["Daily_Demand"]
        else:
            raise ValueError("Daily_Demand_Kg / Daily_Demand 컬럼을 찾을 수 없습니다.")

    nodes = pd.read_csv(nodes_path)

    return df, nodes


df, nodes_df = load_data()

# -------------------------------------------------
# 2) 사이드바 필터
# -------------------------------------------------
st.sidebar.header("⚙️ 필터")

# 연도
all_years = sorted(df["연도"].unique())
default_years = [y for y in all_years if 2024 <= y <= 2030] or all_years
selected_years = st.sidebar.multiselect(
    "연도 선택",
    options=all_years,
    default=default_years,
)
if selected_years:
    df = df[df["연도"].isin(selected_years)]

# 월
all_months = sorted(df["월"].unique())
selected_months = st.sidebar.multiselect(
    "월 선택",
    options=all_months,
    default=all_months,
)
if selected_months:
    df = df[df["월"].isin(selected_months)]

# 요일
weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
existing_weekdays = [d for d in weekday_order if d in df["요일"].unique().tolist()]
selected_weekdays = st.sidebar.multiselect(
    "요일 선택",
    options=existing_weekdays,
    default=existing_weekdays,
)
if selected_weekdays:
    df = df[df["요일"].isin(selected_weekdays)]

# 집계 방식
agg_mode = st.sidebar.radio(
    "집계 방식",
    options=["합계", "평균"],
    index=0,
    horizontal=True,
)
value_col = "Daily_Demand_Kg"
agg_func = "sum" if agg_mode == "합계" else "mean"

if df.empty:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# -------------------------------------------------
# 3) 시도·시군구별 집계 + 좌표 매핑
# -------------------------------------------------
# 시도·시군구별 수요 집계
grouped = (
    df.groupby(["시도", "시군구"], as_index=False)[value_col]
    .agg(agg_func)
    .rename(columns={value_col: "demand_kg"})
)

# Name 컬럼 만들어 all_nodes와 매칭 (예: '서울특별시 강남구')
grouped["Name"] = grouped["시도"].astype(str) + " " + grouped["시군구"].astype(str)

# all_nodes에서 고객 노드만 사용 (Type 없으면 전체 사용)
nodes_customers = nodes_df.copy()
if "Type" in nodes_customers.columns:
    nodes_customers = nodes_customers[nodes_customers["Type"] != "Depot"]

map_df = grouped.merge(
    nodes_customers[["Name", "Lat", "Lng"]],
    on="Name",
    how="left",
)

map_df = map_df.dropna(subset=["Lat", "Lng"])

if map_df.empty:
    st.warning("시군구 수요를 매핑할 좌표 정보를 찾지 못했습니다. all_nodes.csv의 Name / Lat / Lng를 확인해주세요.")
    st.stop()

# 크기/색상 스케일링
d_min = map_df["demand_kg"].min()
d_max = map_df["demand_kg"].max()
if d_max == d_min:
    norm = np.ones(len(map_df))
else:
    norm = (map_df["demand_kg"] - d_min) / (d_max - d_min)

# 점 크기 (수요 많을수록 크게)
map_df["size"] = (norm * 7000) + 2000  # 최소 2000, 최대 9000 정도

# 색상 (수요 많을수록 진한 파랑)
map_df["color_r"] = (50 + norm * 20).astype(int)
map_df["color_g"] = (80 + norm * 50).astype(int)
map_df["color_b"] = (160 + norm * 80).astype(int)

# -------------------------------------------------
# 4) 집계 테이블 + 지도 시각화
# -------------------------------------------------
st.subheader("📊 시도·시군구별 의료폐기물 수요 집계")

st.dataframe(
    grouped.sort_values(["시도", "시군구"]).assign(
        demand_kg=lambda x: x["demand_kg"].round(0).astype(int)
    ),
    use_container_width=True,
)

st.subheader("🗺️ 시도·시군구별 수요 지도")

center_lat = float(map_df["Lat"].mean())
center_lng = float(map_df["Lng"].mean())

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lng,
    zoom=6,
    pitch=0,
)

node_layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df,
    get_position="[Lng, Lat]",
    get_radius="size",
    pickable=True,
    get_fill_color="[color_r, color_g, color_b, 200]",
)

tooltip = {
    "html": (
        "<b>{Name}</b><br/>"
        "시도: {시도}<br/>"
        "시군구: {시군구}<br/>"
        f"{agg_mode} 수요: {{demand_kg}} kg"
    ),
    "style": {"backgroundColor": "white", "color": "black"},
}

deck = pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=view_state,
    layers=[node_layer],
    tooltip=tooltip,
)

st.pydeck_chart(deck, use_container_width=True)

# -------------------------------------------------
# 5) 원본 데이터 일부 미리보기
# -------------------------------------------------
st.subheader("🔍 원본 데이터 미리보기")

with st.expander("cvrp_master_db.csv (현재 필터 적용 상태에서 앞 200행)"):
    st.dataframe(
        df.sort_values(["연도", "월", "요일"]).head(200),
        use_container_width=True,
    )
