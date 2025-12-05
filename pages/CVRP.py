# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import streamlit.components.v1 as components
from pathlib import Path

# -------------------------------------------------
# 1. 페이지 기본 설정
# -------------------------------------------------
st.set_page_config(
    page_title="의료폐기물 수요 및 최적 경로 대시보드",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏥 의료폐기물 수요 모니터링 & 최적 경로 (CVRP)")
st.markdown("---")

# -------------------------------------------------
# 2. 데이터 로드 함수 (캐싱 적용)
# -------------------------------------------------
@st.cache_data
def load_data():
    """
    기본 데이터(cvrp_master_db.csv)와 노드 정보(all_nodes.csv),
    그리고 분석 파이프라인에서 생성된 예측 결과(2025_regional_forecast_final.csv)를 로드합니다.
    """
    data_dir = Path(".")  # 데이터 경로
    
    # 1) 마스터 DB 로드
    cvrp_path = data_dir / "cvrp_master_db.csv"
    if cvrp_path.exists():
        df = pd.read_csv(cvrp_path)
        # 컬럼명 통일 (Daily_Demand_Kg 우선)
        if "Daily_Demand_Kg" not in df.columns:
            if "Daily_Demand" in df.columns:
                df["Daily_Demand_Kg"] = df["Daily_Demand"]
            else:
                df["Daily_Demand_Kg"] = 0
    else:
        st.error(f"❌ '{cvrp_path.resolve()}' 파일을 찾을 수 없습니다.")
        return None, None, None

    # 2) 노드(위경도) 로드
    nodes_path = data_dir / "all_nodes.csv"
    nodes = pd.DataFrame()
    if nodes_path.exists():
        nodes = pd.read_csv(nodes_path)

    # 3) 예측 결과 로드
    forecast_path = data_dir / "2025_regional_forecast_final.csv"
    forecast_df = pd.DataFrame()
    if forecast_path.exists():
        try:
            forecast_df = pd.read_csv(forecast_path, encoding="cp949")
        except:
            try:
                forecast_df = pd.read_csv(forecast_path, encoding="utf-8")
            except:
                pass
    
    return df, nodes, forecast_df

# 데이터 로딩
df_original, nodes_df, forecast_df = load_data()

if df_original is None:
    st.stop()

# -------------------------------------------------
# 3. 사이드바 필터
# -------------------------------------------------
st.sidebar.header("🔍 분석 필터")

# (1) 연도 필터
all_years = sorted(df_original["연도"].unique())
if all_years:
    default_years = [y for y in all_years if y >= 2023] or all_years 
    selected_years = st.sidebar.multiselect("연도 선택", all_years, default=default_years)
    if selected_years:
        df = df_original[df_original["연도"].isin(selected_years)]
    else:
        df = df_original.copy()
else:
    df = df_original.copy()

# (2) 월 필터
all_months = sorted(df["월"].unique())
selected_months = st.sidebar.multiselect("월 선택", all_months, default=all_months)
if selected_months:
    df = df[df["월"].isin(selected_months)]

# (3) 요일 필터
weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
existing_weekdays = [d for d in weekday_order if d in df["요일"].unique().tolist()]
selected_weekdays = st.sidebar.multiselect("요일 선택", existing_weekdays, default=existing_weekdays)
if selected_weekdays:
    df = df[df["요일"].isin(selected_weekdays)]

# (4) 시도 필터
all_sido = sorted(df["시도"].unique())
selected_sido = st.sidebar.multiselect("지역(시도) 선택", all_sido, default=all_sido)
if selected_sido:
    df = df[df["시도"].isin(selected_sido)]

# (5) 집계 기준
agg_mode = st.sidebar.radio("집계 기준", ["합계 (Total)", "평균 (Mean)"], horizontal=True)
agg_func = "sum" if "합계" in agg_mode else "mean"
value_col = "Daily_Demand_Kg"

# -------------------------------------------------
# 4. 데이터 전처리 및 KPI 계산
# -------------------------------------------------
if df.empty:
    st.warning("조건에 맞는 데이터가 없습니다. 필터를 조정해주세요.")
    st.stop()

# KPI 지표
total_demand = df[value_col].sum()
avg_demand = df[value_col].mean()
top_region = df.groupby("시도")[value_col].sum().idxmax()
top_region_val = df.groupby("시도")[value_col].sum().max()

# 시도/시군구 그룹화
grouped = (
    df.groupby(["시도", "시군구"], as_index=False)[value_col]
    .agg(agg_func)
    .rename(columns={value_col: "demand_kg"})
)
grouped["Name"] = grouped["시도"].astype(str) + " " + grouped["시군구"].astype(str)

# 좌표 매핑
if not nodes_df.empty:
    nodes_customers = nodes_df[nodes_df["Type"] != "Depot"] if "Type" in nodes_df.columns else nodes_df
    map_df = grouped.merge(nodes_customers[["Name", "Lat", "Lng"]], on="Name", how="left").dropna(subset=["Lat", "Lng"])
else:
    map_df = pd.DataFrame()

# -------------------------------------------------
# 5. 메인 대시보드 UI
# -------------------------------------------------

# (1) KPI Scorecards
c1, c2, c3, c4 = st.columns(4)
c1.metric("데이터 건수", f"{len(df):,} 건")
c2.metric(f"총 수요량 ({agg_mode})", f"{total_demand:,.0f} kg")
c3.metric(f"평균 수요량", f"{avg_demand:,.1f} kg")
c4.metric("최다 배출 지역", f"{top_region}", f"{top_region_val:,.0f} kg")

st.markdown("###")

# 탭 구성: 지도, 통계, 예측, 경로(New!)
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ 지리적 분포 (Map)", 
    "📊 상세 통계 (Statistics)", 
    "📈 2025 예측 (Forecast)", 
    "🚚 최적 경로 (CVRP Route)"
])

# === TAB 1: 지리적 분포 (PyDeck) ===
with tab1:
    col_map, col_list = st.columns([3, 1])
    
    with col_map:
        if not map_df.empty:
            map_type = st.radio("지도 스타일", ["Scatter Plot (원형)", "Heatmap (밀집도)"], horizontal=True)
            
            view_state = pdk.ViewState(
                latitude=map_df["Lat"].mean(),
                longitude=map_df["Lng"].mean(),
                zoom=6.5,
                pitch=30 if map_type == "Scatter Plot" else 0,
            )

            layers = []
            if "Scatter" in map_type:
                max_val = map_df["demand_kg"].max()
                map_df["radius"] = map_df["demand_kg"] / max_val * 10000 + 1000
                
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position="[Lng, Lat]",
                    get_radius="radius",
                    get_fill_color="[200, 30, 0, 160]",
                    pickable=True,
                    auto_highlight=True,
                )
                layers.append(layer)
                tooltip = {"html": "<b>{Name}</b><br>수요량: {demand_kg} kg"}
            else:
                layer = pdk.Layer(
                    "HeatmapLayer",
                    data=map_df,
                    get_position="[Lng, Lat]",
                    get_weight="demand_kg",
                    radiusPixels=50,
                )
                layers.append(layer)
                tooltip = None

            deck = pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=layers,
                tooltip=tooltip
            )
            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.warning("지도에 표시할 좌표 데이터가 부족합니다.")

    with col_list:
        st.write("📋 **지역별 순위 Top 10**")
        top_10 = grouped.sort_values("demand_kg", ascending=False).head(10)
        st.dataframe(
            top_10[["시도", "시군구", "demand_kg"]].style.format({"demand_kg": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True
        )

# === TAB 2: 통계 차트 (Plotly) ===
with tab2:
    chart1, chart2 = st.columns(2)
    with chart1:
        st.subheader("📍 시도별 수요량 비교")
        sido_grp = df.groupby("시도")[value_col].sum().reset_index()
        fig_bar = px.bar(sido_grp, x="시도", y=value_col, color=value_col, 
                         color_continuous_scale="Reds", title="지역별 총 수요량")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with chart2:
        st.subheader("📅 월별 계절성 패턴")
        month_grp = df.groupby("월")[value_col].mean().reset_index()
        fig_line = px.line(month_grp, x="월", y=value_col, markers=True, 
                           title="월별 평균 수요량 변화", line_shape="spline")
        st.plotly_chart(fig_line, use_container_width=True)

# === TAB 3: 2025 예측 (Forecast) ===
with tab3:
    st.subheader("🔮 2025년 지역별 의료폐기물 발생량 예측")
    
    if not forecast_df.empty:
        # 컬럼 이름 정규화
        rename_map = {c: "최근실적" for c in forecast_df.columns if c.startswith("최근실적")}
        forecast_df = forecast_df.rename(columns=rename_map)

        if "증감률(%)" in forecast_df.columns:
            forecast_df["Status"] = forecast_df["증감률(%)"].apply(
                lambda x: "🔴 급증" if x > 10 else ("🟠 증가" if x > 0 else "🟢 감소/유지")
            )
        
        fig_forecast = px.bar(
            forecast_df.sort_values("2025_예측", ascending=False),
            x="시도", 
            y="2025_예측",
            color="Status",
            color_discrete_map={"🔴 급증": "#FF4B4B", "🟠 증가": "#FFAA00", "🟢 감소/유지": "#00CC96"},
            hover_data=["최근실적", "증감률(%)", "사용모델"],
            title="2025년 시도별 예측 발생량 (Auto-Selected Model 기반)"
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        with st.expander("📄 예측 결과 상세 데이터 보기", expanded=True):
            format_dict = {"2025_예측": "{:,.1f}", "증감률(%)": "{:+.2f}%"}
            if "최근실적" in forecast_df.columns:
                format_dict["최근실적"] = "{:,.1f}"
            st.dataframe(forecast_df.style.format(format_dict), use_container_width=True)
            
    else:
        st.warning("⚠️ 예측 결과 파일(2025_regional_forecast_final.csv)이 없습니다.")

# === TAB 4: 최적 경로 (CVRP Route) ===
with tab4:
    st.subheader("🚛 Folium CVRP 경로 시각화")
    
    # HTML 파일 경로 설정 (현재 디렉토리 기준)
    html_file_name = "cvrp_geojson_visualization_final.html"
    html_path = Path(".") / html_file_name

    if html_path.exists():
        try:
            # HTML 파일 읽기
            html_str = html_path.read_text(encoding="utf-8")
            
            # Streamlit 컴포넌트로 임베딩 (높이 조절 가능)
            components.html(html_str, height=800, scrolling=True)
            
            # 지도 범례/설명
            with st.expander("ℹ️ 지도 범례 및 설명 (Legend)", expanded=True):
                st.markdown("""
                - **⭐ 검은 별 (Black Star)**: 소각장(Depot) 위치
                - **📍 색깔 점 (Colored Markers)**: 수거 지점 (클릭 시 차량 ID 및 적재량 확인 가능)
                - **➖ 색깔 선 (Colored Polyline)**: 차량별 최적 이동 경로 (도로망 기반)
                """)
                
        except Exception as e:
            st.error(f"HTML 파일을 불러오는 중 오류가 발생했습니다: {e}")
    else:
        st.warning(f"⚠️ 경로 시각화 파일({html_file_name})을 찾을 수 없습니다. 경로 최적화 코드를 먼저 실행해주세요.")
