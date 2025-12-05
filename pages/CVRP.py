# -*- coding: utf-8 -*-
"""
2페이지: 의료폐기물 수요 모니터링 + 2025 예측 + CVRP 경로 결과 요약
- 탭(tab) 제거, 섹션별로 세로로 나열
- 고위험군(서울/경기/부산) vs 일반지역 비교, 2025 예측, CVRP 시나리오 반영
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# -------------------------------------------------
# 1. 페이지 기본 설정
# -------------------------------------------------
st.set_page_config(
    page_title="의료폐기물 수요 & 경로 요약",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚚 의료폐기물 수요 모니터링 & 동적 경로 결과 요약")
st.caption("• 수요: cvrp_master_db.csv  • 노드: all_nodes.csv  • 예측: 2025_regional_forecast_final.csv")
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
        return None, None, None

    df = pd.read_csv(cvrp_path)

    if "Daily_Demand_Kg" not in df.columns:
        if "Daily_Demand" in df.columns:
            df["Daily_Demand_Kg"] = df["Daily_Demand"]
        else:
            df["Daily_Demand_Kg"] = 0

    # 2) 노드 (위경도)
    nodes_path = data_dir / "all_nodes.csv"
    nodes_df = pd.DataFrame()
    if nodes_path.exists():
        nodes_df = pd.read_csv(nodes_path)

    # 3) 2025 예측 결과
    forecast_path = data_dir / "2025_regional_forecast_final.csv"
    forecast_df = pd.DataFrame()
    if forecast_path.exists():
        for enc in ("cp949", "utf-8", "utf-8-sig"):
            try:
                forecast_df = pd.read_csv(forecast_path, encoding=enc)
                break
            except Exception:
                continue

    return df, nodes_df, forecast_df


df_original, nodes_df, forecast_df = load_data()
if df_original is None:
    st.stop()

# -------------------------------------------------
# 3. 사이드바 필터
# -------------------------------------------------
st.sidebar.header("🔍 수요 분석 필터")

df = df_original.copy()
value_col = "Daily_Demand_Kg"

# (1) 연도
all_years = sorted(df["연도"].unique())
default_years = [y for y in all_years if y >= 2020] or all_years
sel_years = st.sidebar.multiselect("연도 선택", all_years, default=default_years)
if sel_years:
    df = df[df["연도"].isin(sel_years)]

# (2) 월
all_months = sorted(df["월"].unique())
sel_months = st.sidebar.multiselect("월 선택", all_months, default=all_months)
if sel_months:
    df = df[df["월"].isin(sel_months)]

# (3) 요일
weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
exist_weekdays = [d for d in weekday_order if d in df["요일"].unique().tolist()]
sel_weekdays = st.sidebar.multiselect("요일 선택", exist_weekdays, default=exist_weekdays)
if sel_weekdays:
    df = df[df["요일"].isin(sel_weekdays)]

# (4) 시도
all_sido = sorted(df["시도"].unique())
sel_sido = st.sidebar.multiselect("지역(시도) 선택", all_sido, default=all_sido)
if sel_sido:
    df = df[df["시도"].isin(sel_sido)]

# (5) 집계 기준
agg_mode = st.sidebar.radio("집계 기준", ["합계 (Total)", "평균 (Mean)"], index=0, horizontal=True)
agg_func = "sum" if "합계" in agg_mode else "mean"

if df.empty:
    st.warning("조건에 맞는 데이터가 없습니다. 사이드바 필터를 조정해주세요.")
    st.stop()

# -------------------------------------------------
# 4. 전체 수요 KPI & 전국 패턴 요약
# -------------------------------------------------
st.markdown("## 1. 전국 의료폐기물 수요 요약")

total_demand = df[value_col].sum()
avg_demand = df[value_col].mean()

# 시도별 합계 기준 (고위험군 분석에도 사용)
by_sido_sum = (
    df.groupby("시도", as_index=False)[value_col]
    .sum()
    .rename(columns={value_col: "total_kg"})
)

top_region_row = by_sido_sum.sort_values("total_kg", ascending=False).iloc[0]
top_region = top_region_row["시도"]
top_region_val = top_region_row["total_kg"]

# Top3 시도 비중 (집중도 지표)
top3 = by_sido_sum.sort_values("total_kg", ascending=False).head(3)
top3_share = top3["total_kg"].sum() / by_sido_sum["total_kg"].sum() * 100

# 평일 vs 주말 패턴
weekday_mask = df["요일"].isin(["Mon", "Tue", "Wed", "Thu", "Fri"])
weekend_mask = df["요일"].isin(["Sat", "Sun"])
weekday_mean = df.loc[weekday_mask, value_col].mean()
weekend_mean = df.loc[weekend_mask, value_col].mean() if weekend_mask.any() else np.nan

c1, c2, c3, c4 = st.columns(4)
c1.metric("데이터 건수", f"{len(df):,} 건")
c2.metric(f"총 수요량 ({agg_mode})", f"{total_demand:,.0f} kg")
c3.metric("평일 평균 수요량", f"{weekday_mean:,.1f} kg")
c4.metric("최다 배출 시도", f"{top_region}", f"{top_region_val:,.0f} kg")

st.caption(
    f"※ 상위 3개 시도({', '.join(top3['시도'])})가 전체 수요의 약 **{top3_share:.1f}%**를 차지하며, "
    "발표 자료에서 정의한 고위험군 선정 근거가 됩니다."
)

# --- 1-1. 시도·시군구 기준 지리적 분포 (PyDeck) ---
st.markdown("### 1-1. 시도·시군구 기준 지리적 분포")

# 시도·시군구 그룹
grouped = (
    df.groupby(["시도", "시군구"], as_index=False)[value_col]
    .agg(agg_func)
    .rename(columns={value_col: "demand_kg"})
)
grouped["Name"] = grouped["시도"].astype(str) + " " + grouped["시군구"].astype(str)

if not nodes_df.empty:
    nodes_customers = nodes_df[nodes_df["Type"] != "Depot"] if "Type" in nodes_df.columns else nodes_df
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

# --- 1-2. 월·요일 패턴 ---
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
    wd_grp["요일"] = pd.Categorical(wd_grp["요일"], categories=weekday_order, ordered=True)
    wd_grp = wd_grp.sort_values("요일")
    fig_wd = px.bar(
        wd_grp,
        x="요일",
        y=value_col,
        title="요일별 평균 수요량 (평일 vs 주말 효과)",
    )
    st.plotly_chart(fig_wd, use_container_width=True)

st.markdown(
    """
**해석 포인트**  
- 월·요일별 수요 패턴은 **배차 전략(요일/계절별 차량 수 조절)**의 근거입니다.  
- 특히 발표 슬라이드의 **2030년 4월 월요일 시나리오**는, 과거 4월·월요일 패턴을 기반으로 한 고수요 상황을 대표합니다.
"""
)

# -------------------------------------------------
# 5. 고위험군(서울·경기·부산) vs 일반지역 비교
# -------------------------------------------------
st.markdown("## 2. 고위험군 vs 일반지역 수요 구조 비교")

HIGH_RISK_SIDO = ["서울", "경기", "부산"]

cluster_df = by_sido_sum.copy()
cluster_df["cluster"] = np.where(
    cluster_df["시도"].isin(HIGH_RISK_SIDO),
    "고위험군(서울·경기·부산)",
    "일반지역",
)

# 시도 개수까지 같이 집계
cluster_summary = (
    cluster_df.groupby("cluster", as_index=False)
    .agg({"total_kg": "sum", "시도": "nunique"})
)

# 컬럼명 정리
cluster_summary = cluster_summary.rename(
    columns={"total_kg": "총수요_kg", "시도": "시도수"}
)

cluster_summary["시도당_평균수요_kg"] = cluster_summary["총수요_kg"] / cluster_summary["시도수"]
cluster_summary["비중(%)"] = (
    cluster_summary["총수요_kg"] / cluster_summary["총수요_kg"].sum() * 100
)


c1, c2 = st.columns([1.5, 1])

with c1:
    fig_cluster = px.bar(
        cluster_summary,
        x="cluster",
        y="총수요_kg",
        text=cluster_summary["비중(%)"].map(lambda x: f"{x:.1f}%"),
        title="고위험군 vs 일반지역 총 수요 비교",
        color="cluster",
        color_discrete_sequence=["#ff4b4b", "#4b8bff"],
    )
    fig_cluster.update_traces(textposition="outside")
    st.plotly_chart(fig_cluster, use_container_width=True)

with c2:
    st.markdown("#### 🔎 클러스터 요약")
    st.dataframe(
        cluster_summary
        .rename(columns={
            "총수요_kg": "총수요(kg)",
            "시도수": "시도 수",
            "시도당_평균수요_kg": "시도당 평균수요(kg)",
        })
        .style.format({
            "총수요(kg)": "{:,.0f}",
            "시도당 평균수요(kg)": "{:,.0f}",
            "비중(%)": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown(
        """
- 고위험군(서울·경기·부산)은 **시도 수는 3개에 불과하지만, 전국 수요의 큰 비중**을 차지합니다.  
- 시도당 평균 수요 또한 일반지역에 비해 높은 수준으로,  
  **동일한 차량 1대를 투입했을 때 기대 수거량이 더 큰 구간**임을 의미합니다.  
- 따라서 CVRP에서 이 클러스터를 우선적으로 커버하도록 가중치를 부여했습니다.
        """
    )

# -------------------------------------------------
# 6. 2025년 시도별 예측 결과 요약
# -------------------------------------------------
st.markdown("## 3. 2025년 시도별 의료폐기물 발생량 예측")

if forecast_df is not None and not forecast_df.empty:

    # 최근실적 컬럼 이름 정규화
    for c in list(forecast_df.columns):
        if "최근" in c and "실적" in c:
            forecast_df = forecast_df.rename(columns={c: "최근실적"})
            break

    # 상태 레이블
    if "증감률(%)" in forecast_df.columns:
        def status_label(x):
            try:
                v = float(x)
            except Exception:
                return "🟢 감소/유지"
            if v > 10:
                return "🔴 급증"
            elif v > 0:
                return "🟠 증가"
            else:
                return "🟢 감소/유지"

        forecast_df["Status"] = forecast_df["증감률(%)"].apply(status_label)
    else:
        forecast_df["Status"] = "정보 없음"

    # 막대 차트
    if "2025_예측" in forecast_df.columns and "시도" in forecast_df.columns:
        fig_fc = px.bar(
            forecast_df.sort_values("2025_예측", ascending=False),
            x="시도",
            y="2025_예측",
            color="Status",
            color_discrete_map={
                "🔴 급증": "#FF4B4B",
                "🟠 증가": "#FFAA00",
                "🟢 감소/유지": "#00CC96",
                "정보 없음": "#888888",
            },
            hover_data=[c for c in forecast_df.columns if c not in ["Status"]],
            title="2025년 시도별 예측 발생량 (AutoML 선정 모델 기준)",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    # 고위험군 vs 일반지역: 예측 관점에서 다시 비교
    if {"시도", "2025_예측"}.issubset(forecast_df.columns):
        fc_cluster = forecast_df[["시도", "2025_예측"]].copy()
        fc_cluster["cluster"] = np.where(
            fc_cluster["시도"].isin(HIGH_RISK_SIDO),
            "고위험군(서울·경기·부산)",
            "일반지역",
        )
        fc_summary = (
            fc_cluster.groupby("cluster", as_index=False)["2025_예측"]
            .sum()
        )

        # groupby 결과가 DataFrame 형태인지 확인하고 컬럼명 통일
        if "2025_예측" in fc_summary.columns:
            fc_summary = fc_summary.rename(columns={"2025_예측": "총예측_kg"})
        else:
            # Series 형태일 수 있어서 한 번 더 방어
            fc_summary = fc_summary.to_frame(name="총예측_kg")

        fc_summary["비중(%)"] = (
            fc_summary["총예측_kg"] / fc_summary["총예측_kg"].sum() * 100
        )

        col_fc1, col_fc2 = st.columns([1.5, 1])

        with col_fc1:
            fig_fc_cluster = px.bar(
                fc_summary,
                x="cluster",
                y="총예측_kg",
                text=fc_summary["비중(%)"].map(lambda x: f"{x:.1f}%"),
                title="2025년 예측 기준 고위험군 vs 일반지역",
                color="cluster",
                color_discrete_sequence=["#ff4b4b", "#4b8bff"],
            )
            fig_fc_cluster.update_traces(textposition="outside")
            st.plotly_chart(fig_fc_cluster, use_container_width=True)

        with col_fc2:
            st.markdown("#### 🔁 예측 기준 클러스터 비중")
            st.dataframe(
                fc_summary
                .rename(columns={"총예측_kg": "총예측(kg)"})
                .style.format({"총예측(kg)": "{:,.0f}", "비중(%)": "{:.1f}%"}),
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(
                """
- 2025년 예측에서도 고위험군의 비중은 크게 감소하지 않으며,  
  **향후에도 서울·경기·부산 중심의 수거/소각 인프라 확충이 필요**함을 시사합니다.
                """
            )

    # 상위/하위 지역 요약
    col_hi, col_lo = st.columns(2)

    with col_hi:
        st.markdown("#### 🔴 예측 급증 지역 Top 3")
        if "증감률(%)" in forecast_df.columns:
            top_up = forecast_df.sort_values("증감률(%)", ascending=False).head(3)
            st.dataframe(
                top_up[["시도", "2025_예측", "증감률(%)", "사용모델"]]
                .style.format({"2025_예측": "{:,.1f}", "증감률(%)": "{:+.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("증감률(%) 컬럼이 없어 급증 지역을 계산할 수 없습니다.")

    with col_lo:
        st.markdown("#### 🟢 감소/안정 지역 Top 3")
        if "증감률(%)" in forecast_df.columns:
            bottom = forecast_df.sort_values("증감률(%)", ascending=True).head(3)
            st.dataframe(
                bottom[["시도", "2025_예측", "증감률(%)", "사용모델"]]
                .style.format({"2025_예측": "{:,.1f}", "증감률(%)": "{:+.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("증감률(%) 컬럼이 없어 안정/감소 지역을 계산할 수 없습니다.")

    st.markdown(
        """
- 예측 모델 비교 결과, **LSTM이 가장 낮은 RMSE를 기록하여 최종 선택**되었고  
  (ETS / ARIMA / RandomForest / Prophet 대비 우수)  
- 이 섹션의 수치는 그 **LSTM 기반 예측값**을 바탕으로 합니다.
        """
    )
else:
    st.warning("⚠️ 2025 예측 결과 파일(2025_regional_forecast_final.csv)을 찾을 수 없어, 예측 섹션을 생략합니다.")

# -------------------------------------------------
# 7. CVRP 경로 결과 시각화 (사전 계산된 HTML)
# -------------------------------------------------
st.markdown("## 4. 동적 경로 최적화 결과 (CVRP)")

st.markdown(
    """
발표 자료의 **“2030년 4월 월요일” 시나리오**에서 사용한 것과 동일한  
CVRP 결과 지도를 아래에 임베딩했습니다.  

- 전국 수요 분포 및 고위험군(서울·경기·부산)을 고려한 **다중 소각장·다차량 경로**  
- **총 처리 물량, 차량 수, 운행 거리, 총 비용**은 발표 슬라이드와 동일한 가정 하에서 계산된 값입니다.
"""
)

html_file_name = "cvrp_geojson_visualization_final.html"
html_path = Path(".") / html_file_name

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
1) **수요 예측 결과**  
2) **고위험군 우선 수거 패널티(서울·경기·부산)**  
3) **차량 용량·고정비·변동비**  
를 동시에 고려해 산출된 결과입니다.
                """
            )
    except Exception as e:
        st.error(f"경로 HTML 파일을 임베딩하는 중 오류가 발생했습니다: {e}")
else:
    st.warning(f"⚠️ '{html_file_name}' 파일을 찾을 수 없습니다. 경로 최적화 스크립트를 먼저 실행해 주세요.")

# -------------------------------------------------
# 8. 원본 데이터 미리보기 (선택 사항)
# -------------------------------------------------
with st.expander("🔍 원본 수요 데이터 미리보기 (필터 적용 후 상위 200행)", expanded=False):
    st.dataframe(
        df.sort_values(["연도", "월", "요일"]).head(200),
        use_container_width=True,
    )

# -------------------------------------------------
# 9. 자동 인사이트 요약 (발표용 문장)
# -------------------------------------------------
st.markdown("---")
st.markdown("## 🧾 자동 인사이트 요약")

insights = []

# 고위험군 관련
if not cluster_summary.empty:
    high_row = cluster_summary[cluster_summary["cluster"].str.contains("고위험군")].iloc[0]
    low_row = cluster_summary[cluster_summary["cluster"].str.contains("일반지역")].iloc[0]
    insights.append(
        f"- **고위험군(서울·경기·부산)**은 전체 시도의 일부(3개)에 불과하지만, "
        f"전국 의료폐기물 수요의 약 **{high_row['비중(%)']:.1f}%**를 차지합니다."
    )
    ratio_mean = high_row["시도당_평균수요_kg"] / low_row["시도당_평균수요_kg"]
    insights.append(
        f"- 시도당 평균 수요 기준으로 보면, 고위험군은 일반지역 대비 약 **{ratio_mean:.1f}배** 높은 수준입니다."
    )

# 평일/주말 차이
if not np.isnan(weekday_mean) and not np.isnan(weekend_mean):
    diff = weekday_mean - weekend_mean
    direction = "높습니다" if diff > 0 else "낮습니다"
    insights.append(
        f"- 평일 평균 수요는 **{weekday_mean:,.1f} kg**, 주말은 **{weekend_mean:,.1f} kg**로, "
        f"평일이 주말보다 약 **{abs(diff):,.1f} kg** {direction}."
    )

# 예측 데이터 기반
if forecast_df is not None and not forecast_df.empty and {"시도", "2025_예측"}.issubset(forecast_df.columns):
    fc_cluster = forecast_df[["시도", "2025_예측"]].copy()
    fc_cluster["cluster"] = np.where(
        fc_cluster["시도"].isin(HIGH_RISK_SIDO),
        "고위험군",
        "일반지역",
    )
    fc_summary = (
        fc_cluster.groupby("cluster", as_index=False)["2025_예측"]
        .sum()
        .rename(columns={"2025_예측": "총예측"})
    )
    if len(fc_summary) == 2:
        high_fc = fc_summary[fc_summary["cluster"] == "고위험군"]["총예측"].iloc[0]
        low_fc = fc_summary[fc_summary["cluster"] == "일반지역"]["총예측"].iloc[0]
        share_fc = high_fc / (high_fc + low_fc) * 100
        insights.append(
            f"- 2025년 예측 기준으로도 고위험군(서울·경기·부산)은 전체 예측 수요의 약 **{share_fc:.1f}%**를 유지하여, "
            "향후에도 집중 관리가 필요한 권역으로 남을 가능성이 높습니다."
        )

if insights:
    for line in insights:
        st.markdown(line)
else:
    st.write("추가로 요약할 인사이트를 찾지 못했습니다. 데이터 컬럼 구성을 확인해 주세요.")
