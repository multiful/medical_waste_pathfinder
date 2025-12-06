# -*- coding: utf-8 -*-
"""
2페이지: 의료폐기물 수요 모니터링 + CVRP 경로 결과 요약
- 밝은 테마 + streamlit-option-menu 사이드 메뉴
- 고위험군(서울/경기/부산) vs 일반지역 비교
- CVRP 결과 지도 임베딩
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu

from ui_theme import apply_theme

# -------------------------------------------------
# 1. 페이지 기본 설정 & 테마
# -------------------------------------------------
st.set_page_config(
    page_title="의료폐기물 수요 모니터링 & 경로 결과",
    page_icon="🚚",
    layout="wide",
)
apply_theme()

st.title("🚚 의료폐기물 수요 모니터링 & 동적 경로 결과 요약")
st.markdown("---")

# -------------------------------------------------
# 2. 데이터 로드 (캐싱)
# -------------------------------------------------
@st.cache_data
def load_data():
    data_dir = Path("./data")

    # 1) 수요 마스터 DB
    cvrp_path = data_dir / "cvrp_master_db.csv"
    if not cvrp_path.exists():
        st.error(f"❌ '{cvrp_path.resolve()}' 파일을 찾을 수 없습니다.")
        return None, None

    try:
        try:
            df = pd.read_csv(cvrp_path, encoding="cp949")
        except UnicodeDecodeError:
            df = pd.read_csv(cvrp_path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        st.error(
            f"❌ '{cvrp_path.name}' 파일이 비어 있습니다.\n"
            "로컬에서 cvrp_master_db.csv 내용을 확인하고, "
            "데이터가 들어있는 파일로 다시 업로드/커밋해 주세요."
        )
        return None, None

    if "Daily_Demand_Kg" not in df.columns:
        if "Daily_Demand" in df.columns:
            df["Daily_Demand_Kg"] = df["Daily_Demand"]
        else:
            df["Daily_Demand_Kg"] = 0

    # 2) 노드 (위경도)
    nodes_path = data_dir / "all_nodes.csv"
    nodes_df = pd.DataFrame()
    if nodes_path.exists():
        try:
            nodes_df = pd.read_csv(nodes_path, encoding="cp949")
        except UnicodeDecodeError:
            nodes_df = pd.read_csv(nodes_path, encoding="utf-8-sig")

    return df, nodes_df


df_original, nodes_df = load_data()
if df_original is None:
    st.stop()

value_col = "Daily_Demand_Kg"
weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# -------------------------------------------------
# 3. 사이드바: 메뉴 + 필터
# -------------------------------------------------
all_years = sorted(df_original["연도"].unique())
default_years = [y for y in all_years if y >= 2020] or all_years
all_months = sorted(df_original["월"].unique())
all_sido = sorted(df_original["시도"].unique())
exist_weekdays = [d for d in weekday_order if d in df_original["요일"].unique().tolist()]

with st.sidebar:
    selected_menu = option_menu(
        "Menu",
        ["요약", "수요 패턴", "고위험군 vs 일반", "CVRP 경로"],
        icons=["speedometer2", "bar-chart-line-fill", "people-fill", "truck"],
        menu_icon="caret-down-fill",
        default_index=0,
        styles={
            "container": {
                "padding": "0.5rem 0.5rem",
                "background-color": "#fafafa",
            },
            "icon": {"color": "#2563eb", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "2px 0",
                "--hover-color": "#e5edff",
            },
            "nav-link-selected": {
                "background-color": "#2563eb",
                "color": "white",
                "font-weight": "600",
            },
        },
    )

    st.markdown("---")
    st.markdown("### 🔍 수요 분석 필터")

    sel_years = st.multiselect("연도 선택", all_years, default=default_years)
    sel_months = st.multiselect("월 선택", all_months, default=all_months)
    sel_weekdays = st.multiselect("요일 선택", exist_weekdays, default=exist_weekdays)
    sel_sido = st.multiselect("지역(시도) 선택", all_sido, default=all_sido)

    agg_mode = st.radio(
        "집계 기준",
        ["합계 (Total)", "평균 (Mean)"],
        index=0,
        horizontal=True,
    )

# 필터 적용
df = df_original.copy()

if sel_years:
    df = df[df["연도"].isin(sel_years)]
if sel_months:
    df = df[df["월"].isin(sel_months)]
if sel_weekdays:
    df = df[df["요일"].isin(sel_weekdays)]
if sel_sido:
    df = df[df["시도"].isin(sel_sido)]

if df.empty:
    st.warning("조건에 맞는 데이터가 없습니다. 사이드바 필터를 조정해주세요.")
    st.stop()

agg_func = "sum" if "합계" in agg_mode else "mean"

# -------------------------------------------------
# 4. 공통 지표 계산 (여러 메뉴에서 재사용)
# -------------------------------------------------
total_demand = df[value_col].sum()
avg_demand = df[value_col].mean()

by_sido_sum = (
    df.groupby("시도", as_index=False)[value_col]
    .sum()
    .rename(columns={value_col: "total_kg"})
)
top_region_row = by_sido_sum.sort_values("total_kg", ascending=False).iloc[0]
top_region = top_region_row["시도"]
top_region_val = top_region_row["total_kg"]

top3 = by_sido_sum.sort_values("total_kg", ascending=False).head(3)
top3_share = top3["total_kg"].sum() / by_sido_sum["total_kg"].sum() * 100

weekday_mask = df["요일"].isin(["Mon", "Tue", "Wed", "Thu", "Fri"])
weekend_mask = df["요일"].isin(["Sat", "Sun"])
weekday_mean = df.loc[weekday_mask, value_col].mean()
weekend_mean = df.loc[weekend_mask, value_col].mean() if weekend_mask.any() else np.nan

HIGH_RISK_SIDO = ["서울", "경기", "부산"]
cluster_df = by_sido_sum.copy()
cluster_df["cluster"] = np.where(
    cluster_df["시도"].isin(HIGH_RISK_SIDO),
    "고위험군(서울·경기·부산)",
    "일반지역",
)
cluster_summary = (
    cluster_df.groupby("cluster", as_index=False)
    .agg({"total_kg": "sum", "시도": "nunique"})
    .rename(columns={"total_kg": "총수요_kg", "시도": "시도수"})
)
cluster_summary["시도당_평균수요_kg"] = (
    cluster_summary["총수요_kg"] / cluster_summary["시도수"]
)
cluster_summary["비중(%)"] = (
    cluster_summary["총수요_kg"] / cluster_summary["총수요_kg"].sum() * 100
)

# -------------------------------------------------
# 5-1. 메뉴: 요약
# -------------------------------------------------
if selected_menu == "요약":
    st.markdown("## 1. 전국 의료폐기물 수요 요약")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("데이터 건수", f"{len(df):,} 건")
    c2.metric(f"총 수요량 ({agg_mode})", f"{total_demand:,.0f} kg")
    c3.metric("평일 평균 수요량", f"{weekday_mean:,.1f} kg")
    c4.metric("최다 배출 시도", f"{top_region}", f"{top_region_val:,.0f} kg")

    st.caption(
        f"※ 상위 3개 시도({', '.join(top3['시도'])})가 전체 수요의 약 **{top3_share:.1f}%**를 차지합니다."
    )

    # 인사이트 텍스트
    st.markdown("### 🧾 자동 인사이트 요약")

    insights = []
    if not cluster_summary.empty:
        high_row = cluster_summary[
            cluster_summary["cluster"].str.contains("고위험군")
        ].iloc[0]
        low_row = cluster_summary[
            cluster_summary["cluster"].str.contains("일반지역")
        ].iloc[0]
        insights.append(
            f"- **고위험군(서울·경기·부산)**은 전체 시도의 일부(3개)에 불과하지만, "
            f"전국 의료폐기물 수요의 약 **{high_row['비중(%)']:.1f}%**를 차지합니다."
        )
        ratio_mean = (
            high_row["시도당_평균수요_kg"] / low_row["시도당_평균수요_kg"]
        )
        insights.append(
            f"- 시도당 평균 수요 기준으로 보면, 고위험군은 일반지역 대비 약 **{ratio_mean:.1f}배** 높은 수준입니다."
        )

    if not np.isnan(weekday_mean) and not np.isnan(weekend_mean):
        diff = weekday_mean - weekend_mean
        direction = "높습니다" if diff > 0 else "낮습니다"
        insights.append(
            f"- 평일 평균 수요는 **{weekday_mean:,.1f} kg**, 주말은 **{weekend_mean:,.1f} kg**로, "
            f"평일이 주말보다 약 **{abs(diff):,.1f} kg** {direction}."
        )

    if insights:
        for line in insights:
            st.markdown(line)
    else:
        st.write("요약할 인사이트를 찾지 못했습니다.")

# -------------------------------------------------
# 5-2. 메뉴: 수요 패턴 (지도 + 월/요일)
# -------------------------------------------------
if selected_menu == "수요 패턴":
    st.markdown("## 1. 공간·시간 패턴 (지도 + 월/요일)")

    # 시도·시군구 그룹
    grouped = (
        df.groupby(["시도", "시군구"], as_index=False)[value_col]
        .agg(agg_func)
        .rename(columns={value_col: "demand_kg"})
    )
    grouped["Name"] = (
        grouped["시도"].astype(str) + " " + grouped["시군구"].astype(str)
    )

    if not nodes_df.empty:
        nodes_customers = (
            nodes_df[nodes_df["Type"] != "Depot"]
            if "Type" in nodes_df.columns
            else nodes_df
        )
        map_df = grouped.merge(
            nodes_customers[["Name", "Lat", "Lng"]],
            on="Name",
            how="left",
        ).dropna(subset=["Lat", "Lng"])
    else:
        map_df = pd.DataFrame()

    col_map, col_rank = st.columns([3, 1])

    with col_map:
        if not map_df.empty:
            max_val = map_df["demand_kg"].max()
            map_df["radius"] = map_df["demand_kg"] / max_val * 12000 + 1500

            view_state = pdk.ViewState(
                latitude=float(map_df["Lat"].mean()),
                longitude=float(map_df["Lng"].mean()),
                zoom=6.3,
                pitch=30,
            )

            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[Lng, Lat]",
                get_radius="radius",
                get_fill_color="[200, 30, 0, 160]",
                pickable=True,
                auto_highlight=True,
            )

            deck = pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[scatter_layer],
                tooltip={"html": "<b>{Name}</b><br>수요량: {demand_kg} kg"},
            )
            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.info("좌표 정보(all_nodes.csv)가 없어 지도 시각화를 생략합니다.")

    with col_rank:
        st.markdown("#### 📋 지역별 수요 Top 10")
        top10 = grouped.sort_values("demand_kg", ascending=False).head(10)
        st.dataframe(
            top10[["시도", "시군구", "demand_kg"]]
            .rename(columns={"demand_kg": "수요(kg)"})
            .style.format({"수요(kg)": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### 1-2. 월·요일별 계절성 패턴")

    col_m, col_w = st.columns(2)
    with col_m:
        mon_grp = df.groupby("월", as_index=False)[value_col].mean()
        fig_mon = px.line(
            mon_grp,
            x="월",
            y=value_col,
            markers=True,
            title="월별 평균 수요량",
        )
        st.plotly_chart(fig_mon, use_container_width=True)

    with col_w:
        wd_grp = df.groupby("요일", as_index=False)[value_col].mean()
        wd_grp["요일"] = pd.Categorical(
            wd_grp["요일"], categories=weekday_order, ordered=True
        )
        fig_wd = px.bar(
            wd_grp,
            x="요일",
            y=value_col,  # ✅ 이렇게 고쳐야 함
            title="요일별 평균 수요량 (평일 vs 주말 효과)",
        )
        st.plotly_chart(fig_wd, use_container_width=True)

    with st.expander("🔍 원본 수요 데이터 미리보기 (필터 적용 후 상위 200행)", expanded=False):
        st.dataframe(
            df.sort_values(["연도", "월", "요일"]).head(200),
            use_container_width=True,
        )

# -------------------------------------------------
# 5-3. 메뉴: 고위험군 vs 일반
# -------------------------------------------------
if selected_menu == "고위험군 vs 일반":
    st.markdown("## 2. 고위험군(서울·경기·부산) vs 일반지역 비교")

    c1, c2 = st.columns([1.5, 1])

    with c1:
        fig_cluster = px.bar(
            cluster_summary,
            x="cluster",
            y="총수요_kg",
            text=cluster_summary["비중(%)"].map(lambda x: f"{x:.1f}%"),
            title="고위험군 vs 일반지역 총 수요 비교",
            color="cluster",
            color_discrete_sequence=["#f97373", "#4b8bff"],
        )
        fig_cluster.update_traces(textposition="outside")
        st.plotly_chart(fig_cluster, use_container_width=True)

    with c2:
        st.markdown("#### 🔎 클러스터 요약")
        st.dataframe(
            cluster_summary.rename(
                columns={
                    "총수요_kg": "총수요(kg)",
                    "시도수": "시도 수",
                    "시도당_평균수요_kg": "시도당 평균수요(kg)",
                }
            ).style.format(
                {
                    "총수요(kg)": "{:,.0f}",
                    "시도당 평균수요(kg)": "{:,.0f}",
                    "비중(%)": "{:.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            """
- 고위험군(서울·경기·부산)은 시도 수는 3개에 불과하지만, 전국 수요의 큰 비중을 차지합니다.  
- 시도당 평균 수요 또한 일반지역에 비해 높아, **차량 1대를 투입했을 때 기대 수거량이 더 큰 구간**입니다.  
- CVRP 모델에서 이 클러스터에 우선순위를 줘서 경로를 구성했습니다.
            """
        )

# -------------------------------------------------
# 5-4. 메뉴: CVRP 경로 결과
# -------------------------------------------------
if selected_menu == "CVRP 경로":
    st.markdown("## 3. 동적 경로 최적화 결과 (CVRP)")

    st.markdown(
        """
발표 자료의 **“2030년 4월 월요일” 시나리오**에서 사용한 것과 동일한  
CVRP 결과 지도를 아래에 임베딩했습니다.  

- 전국 수요 분포 및 고위험군(서울·경기·부산)을 고려한 **다중 소각장·다차량 경로**  
- **총 처리 물량, 차량 수, 운행 거리, 총 비용**은 발표 슬라이드와 동일한 가정 하에서 계산된 값입니다.
"""
    )

    html_file_name = "cvrp_geojson_visualization_final.html"
    html_path = Path("data") / html_file_name

    if html_path.exists():
        try:
            html_str = html_path.read_text(encoding="utf-8")
            components.html(html_str, height=800, scrolling=True)

            with st.expander("ℹ️ 지도 범례 / 해석 가이드", expanded=True):
                st.markdown(
                    """
- **⭐ 검은 별**: 소각장(Depot) 위치  
- **색깔 점**: 각 차량이 방문하는 수거 지점 (팝업에 차량 ID·적재량 표시)  
- **색깔 선**: 차량별 주행 경로 (요일·월별 수요를 반영한 동적 CVRP 결과)  

이 경로는  
1) **수요 데이터**  
2) **고위험군 우선 수거 패널티(서울·경기·부산)**  
3) **차량 용량·고정비·변동비**  
를 동시에 고려해 산출된 결과입니다.
                    """
                )
        except Exception as e:
            st.error(f"경로 HTML 파일을 임베딩하는 중 오류가 발생했습니다: {e}")
    else:
        st.warning("⚠️ 'data/cvrp_geojson_visualization_final.html' 파일을 찾을 수 없습니다.")
