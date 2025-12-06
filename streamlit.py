# app.py — 의료폐기물 분석 대시보드 (Page 1: final_df 인사이트)
# 데이터: final_df.csv (시도별 의료폐기물 + 병원/의원 수 + 인구/인프라 등)

from pathlib import Path
import json

import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import streamlit as st

from ui_theme import apply_theme

apply_theme()   # 또는 "paper-light", "glass-dark"

def inject_custom_css():
    st.markdown(
        """
        <style>
        /* 전체 컨테이너 폭 & 여백 */
        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* 타이틀 그라데이션 */
        h1 {
            font-size: 2.6rem !important;
            font-weight: 800 !important;
            background: linear-gradient(90deg, #ff4b4b, #fb923c, #facc15);
            -webkit-background-clip: text;
            color: transparent;
        }

        /* 사이드바 배경 */
        [data-testid="stSidebar"] {
            background-color: #020617;
            border-right: 1px solid rgba(148, 163, 184, 0.3);
        }

        /* metric 카드 이쁘게 */
        [data-testid="metric-container"] {
            background-color: #020617;
            border-radius: 0.75rem;
            padding: 1rem 1.2rem;
            border: 1px solid rgba(148, 163, 184, 0.4);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.8);
        }
        [data-testid="metric-container"] > div {
            color: #e5e7eb !important;
        }

        /* expander 스타일 */
        details {
            border-radius: 0.75rem;
            background-color: #020617;
            border: 1px solid rgba(148, 163, 184, 0.4);
        }

        /* 데이터프레임 헤더 */
        .stDataFrame thead tr th {
            background-color: #020617 !important;
            color: #e5e7eb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_custom_css()


# -------------------------------
# 기본 설정
# -------------------------------
st.set_page_config(
    layout="wide",
    page_title="의료폐기물 분석 대시보드",
    page_icon="🧪",
)
alt.data_transformers.disable_max_rows()
st.title("의료폐기물 분석 대시보드")


DATA_FILE = "data/final_df.csv"
GEO_FILE = "data/TL_SCCO_CTPRVN.json"  # 지도 파일 이름

# -------------------------------
# 공용 유틸 함수
# -------------------------------
def series_to_df(s: pd.Series, value_name: str, index_name: str) -> pd.DataFrame:
    s = s.copy()
    df_tmp = s.to_frame(value_name)
    idx_name = index_name if index_name not in df_tmp.columns else f"{index_name}_idx"
    df_tmp = df_tmp.rename_axis(idx_name).reset_index()
    if idx_name != index_name:
        df_tmp = df_tmp.rename(columns={idx_name: index_name})
    return df_tmp


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    # 여러 인코딩을 순서대로 시도 (cp949, utf-8-sig, utf-8)
    encodings = ["cp949", "utf-8-sig", "utf-8"]

    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
    else:
        st.error(f"CSV 인코딩을 해석하지 못했습니다.\n마지막 에러: {last_err}")
        st.stop()

    if "시도" in df.columns:
        df["시도"] = df["시도"].astype(str).str.strip()
    return df

# 시도 → TL_SCCO_CTPRVN.json 의 CTP_KOR_NM 매핑
SIDO_TO_SHP = {
    "서울": "서울특별시",
    "부산": "부산광역시",
    "대구": "대구광역시",
    "인천": "인천광역시",
    "광주": "광주광역시",
    "대전": "대전광역시",
    "울산": "울산광역시",
    "세종": "세종특별자치시",
    "경기": "경기도",
    "강원": "강원도",
    "충북": "충청북도",
    "충남": "충청남도",
    "전북": "전라북도",
    "전남": "전라남도",
    "경북": "경상북도",
    "경남": "경상남도",
    "제주": "제주특별자치도",
}



@st.cache_data(show_spinner=False)
def load_geojson(path: str):
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# -------------------------------
# 데이터 로딩
# -------------------------------
if not Path(DATA_FILE).exists():
    st.error(f"'{DATA_FILE}' 파일을 찾을 수 없습니다. 같은 폴더에 final_df.csv를 두고 다시 실행해 주세요.")
    st.stop()

df_raw = load_data(DATA_FILE)


# 주요 컬럼 이름들
TARGET_COL = "지역별_의료폐기물"
TARGET_TRANS_COL = "지역별_의료폐기물_TRANS"  # 있으면 선택해서 사용
DENTAL_COL = "치과병원"
REHAB_COL = "요양병원"
INFRA_COL = "의료인프라_강도"

FACILITY_HOSP_COLS = [
    "상급종합병원",
    "종합병원",
    "치과병원",
    "한방병원",
    "요양병원",
    "정신병원",
]
FACILITY_CLINIC_COLS = ["의원", "치과의원", "한의원"]

# -------------------------------
# 사이드바 필터
# -------------------------------
with st.sidebar:
    st.header("필터")

    df = df_raw.copy()

    # 연도 필터
    if "연도" in df.columns:
        years = sorted(df["연도"].dropna().unique().tolist())
        sel_years = st.multiselect("연도 선택", options=years, default=years)
        if sel_years:
            df = df[df["연도"].isin(sel_years)]
        st.caption(f"선택된 연도: {', '.join(map(str, sel_years)) if sel_years else '전체'}")
    else:
        st.info("연도 컬럼이 없어 연도 필터는 표시하지 않습니다.")

    # 시도 필터
    if "시도" in df.columns:
        sidos = sorted(df["시도"].dropna().unique().tolist())
        sel_sidos = st.multiselect("시도 선택", options=sidos, default=sidos)
        if sel_sidos:
            df = df[df["시도"].isin(sel_sidos)]
        st.caption(f"선택된 시도: {', '.join(sel_sidos) if sel_sidos else '전체'}")

    # 타깃(원본 vs 변환) 선택
    target_options = []
    if TARGET_COL in df.columns:
        target_options.append(("원본 (지역별_의료폐기물)", TARGET_COL))
    if TARGET_TRANS_COL in df.columns:
        target_options.append(("변환값 (지역별_의료폐기물_TRANS)", TARGET_TRANS_COL))

    if not target_options:
        st.error("의료폐기물 컬럼(지역별_의료폐기물)이 존재하지 않습니다.")
        st.stop()

    label_list = [lbl for lbl, _ in target_options]
    default_idx = 1 if len(target_options) > 1 else 0
    sel_label = st.radio("의료폐기물 지표 선택", label_list, index=default_idx)
    TARGET_USED = dict(target_options)[sel_label]
    st.caption(f"분석 타깃: **{TARGET_USED}**")

# -------------------------------
# 상단 KPI 카드
# -------------------------------
st.subheader("요약 지표")

k1, k2, k3, k4 = st.columns(4)

# KPI는 선택된 타깃 기준으로 계산
target_series = df[TARGET_USED]
total_waste = target_series.sum()

if "시도" in df.columns:
    mean_waste_per_region = df.groupby("시도")[TARGET_USED].sum().mean()
else:
    mean_waste_per_region = np.nan

if DENTAL_COL in df.columns:
    total_dental = df[DENTAL_COL].sum()
    waste_per_dental = total_waste / total_dental if total_dental > 0 else np.nan
else:
    waste_per_dental = np.nan

if "시도" in df.columns:
    top_region = (
        df.groupby("시도")[TARGET_USED]
        .sum()
        .sort_values(ascending=False)
        .head(1)
    )
    top_region_name = top_region.index[0]
    top_region_val = float(top_region.iloc[0])
else:
    top_region_name, top_region_val = "-", np.nan

with k1:
    st.metric("총 의료폐기물 (선택 지표 합계)", f"{total_waste:,.0f}")
with k2:
    if not np.isnan(mean_waste_per_region):
        st.metric("시도별 평균 의료폐기물", f"{mean_waste_per_region:,.0f}")
    else:
        st.metric("시도별 평균 의료폐기물", "N/A")
with k3:
    if not np.isnan(waste_per_dental):
        st.metric("치과병원 1기관당 평균 의료폐기물", f"{waste_per_dental:,.1f}")
    else:
        st.metric("치과병원 1기관당 평균 의료폐기물", "N/A")
with k4:
    st.metric(
        "의료폐기물 최다 배출 시도",
        f"{top_region_name} ({top_region_val:,.0f})" if not np.isnan(top_region_val) else "N/A",
    )

st.caption("※ 변환값(TRANS)을 선택한 경우, 절대량(톤)이 아닌 '상대 지표'로 해석해야 합니다.")
st.markdown("---")

# -------------------------------
# 탭 레이아웃 (시설유형 탭 제거 → 3개 탭만 사용)
# -------------------------------
tab1, tab2, tab3 = st.tabs(
    ["시도별 비교", "상관·회귀 분석", "의료 인프라(SEM 관점)"]
)

# -------------------------------
# Tab1: 시도별 의료폐기물 + 지도
# -------------------------------
with tab1:
    st.markdown("### 시도별 의료폐기물 비교")

    if {"시도", TARGET_USED}.issubset(df.columns):
        grouped = df.groupby("시도", as_index=False).agg(
            의료폐기물=(TARGET_USED, "sum"),
            치과병원=(DENTAL_COL, "sum") if DENTAL_COL in df.columns else ("시도", "size"),
        )
        if DENTAL_COL in df.columns:
            grouped["치과병원_당_폐기물"] = grouped["의료폐기물"] / grouped["치과병원"].replace(0, np.nan)

        c1, c2 = st.columns([2, 1], gap="large")

        with c1:
            base = grouped.sort_values(
                "치과병원_당_폐기물" if "치과병원_당_폐기물" in grouped.columns else "의료폐기물"
            )
            bar = (
                alt.Chart(base)
                .mark_bar()
                .encode(
                    x=alt.X("시도:N", sort=None),
                    y=alt.Y(
                        "치과병원_당_폐기물:Q",
                        title="치과병원 1기관당 의료폐기물(선택 지표)",
                    )
                    if "치과병원_당_폐기물" in base.columns
                    else alt.Y("의료폐기물:Q", title="의료폐기물(선택 지표)"),
                    tooltip=base.columns.tolist(),
                )
                .properties(width="container", height=380)
            )
            st.altair_chart(bar, use_container_width=True)

        with c2:
            line = (
                alt.Chart(grouped)
                .transform_fold(
                    ["의료폐기물", "치과병원"],
                    as_=["지표", "값"],
                )
                .mark_line(point=True)
                .encode(
                    x=alt.X("시도:N", sort=None),
                    y=alt.Y("값:Q", title="값(선택 지표 / 기관수)"),
                    color="지표:N",
                    tooltip=["시도:N", "지표:N", "값:Q"],
                )
                .properties(height=380)
            )
            st.altair_chart(line, use_container_width=True)

        # 지도 시각화
        st.markdown("#### 시도별 의료폐기물 지리적 분포")

        geo_data = load_geojson(GEO_FILE)
        if geo_data is not None:
            map_agg = grouped[["시도", "의료폐기물"]].copy()
            map_agg["CTP_KOR_NM"] = map_agg["시도"].map(SIDO_TO_SHP).fillna(map_agg["시도"])

            fig = px.choropleth(
                map_agg,
                geojson=geo_data,
                locations="CTP_KOR_NM",
                featureidkey="properties.CTP_KOR_NM",
                color="의료폐기물",
                color_continuous_scale="OrRd",
                labels={"의료폐기물": "의료폐기물(선택 지표)"},
                hover_data={"시도": True, "의료폐기물": ":,.0f"},
                title="시도별 의료폐기물 (합계, 선택 지표)",
            )
            fig.update_geos(fitbounds="locations", visible=False)
            fig.update_layout(height=450, margin=dict(l=0, r=0, t=60, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"'{GEO_FILE}' 파일이 없어 지도 시각화를 생략합니다.")

    else:
        st.warning("시도 또는 의료폐기물 컬럼이 없어 시도별 비교를 그릴 수 없습니다.")

# -------------------------------
# Tab2: 상관·회귀 분석
# -------------------------------
with tab2:
    st.markdown("### 의료폐기물과 의료 인프라 지표 간 상관·회귀 분석")

    if TARGET_USED not in df.columns:
        st.warning("선택된 의료폐기물 컬럼이 없어 상관 분석을 수행할 수 없습니다.")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()[TARGET_USED].drop(labels=[TARGET_USED])
        corr_df = corr.sort_values(ascending=False).to_frame("Pearson r")
        corr_df["abs_r"] = corr_df["Pearson r"].abs()
        corr_df = corr_df.sort_values("abs_r", ascending=True)

        st.markdown("**의료폐기물과의 상관계수 (상대적으로 큰 것일수록 영향력 가능성↑)**")
        corr_chart = (
            alt.Chart(corr_df.reset_index())
            .mark_bar()
            .encode(
                x=alt.X("Pearson r:Q"),
                y=alt.Y("index:N", title="변수명", sort="-x"),
                color=alt.Color("Pearson r:Q", scale=alt.Scale(scheme="blueorange")),
                tooltip=["index", "Pearson r"],
            )
            .properties(height=max(280, 18 * len(corr_df)))
        )
        st.altair_chart(corr_chart, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 특정 시설 수 vs 의료폐기물 (산점도 + 회귀선)")

        candidate_xcols = [c for c in FACILITY_HOSP_COLS + FACILITY_CLINIC_COLS if c in df.columns]
        if not candidate_xcols:
            candidate_xcols = [c for c in numeric_cols if c != TARGET_USED]

        sel_x = st.selectbox("x축 변수 선택", options=candidate_xcols, index=0)

        # 로그 스케일 옵션
        col_log1, col_log2 = st.columns(2)
        with col_log1:
            use_log_x = st.checkbox("x축 로그 스케일", value=False)
        with col_log2:
            use_log_y = st.checkbox("y축 로그 스케일", value=False)

        scatter_df = df[[sel_x, TARGET_USED]].dropna()

        x_enc = alt.X(
            f"{sel_x}:Q",
            title=sel_x,
            scale=alt.Scale(type="log") if use_log_x else alt.Undefined,
        )
        y_enc = alt.Y(
            f"{TARGET_USED}:Q",
            title="의료폐기물(선택 지표)",
            scale=alt.Scale(type="log") if use_log_y else alt.Undefined,
        )

        sc = (
            alt.Chart(scatter_df)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=x_enc,
                y=y_enc,
                tooltip=[sel_x, TARGET_USED],
            )
        )

        reg = sc.transform_regression(sel_x, TARGET_USED, method="linear").mark_line(color="orange")

        st.altair_chart(sc + reg, use_container_width=True)
        st.caption("※ 점 하나는 (시도×연도) 단위 하나를 의미. 직선 기울기는 단순 선형회귀 계수에 해당.")

# -------------------------------
# Tab3: 의료 인프라(SEM 관점)
# -------------------------------
with tab3:
    st.markdown("### 의료 인프라 강도와 의료폐기물 (SEM 구조 해석용)")

    if {INFRA_COL, DENTAL_COL, REHAB_COL}.issubset(df.columns) and TARGET_USED in df.columns:
        info_col1, info_col2 = st.columns([2, 1])

        with info_col1:
            st.markdown(
                """
**가설(H4)**  

- 치과병원·요양병원 증가 → 의료인프라 강도 증가  
- 의료인프라 강도 증가 → 의료폐기물 증가 (또는 효율성 효과로 감소)  

이 탭은 위 SEM 구조를 이해하기 위한 기초 EDA를 보여줍니다.
                """
            )

        with info_col2:
            # 간단 요약 메트릭
            mean_infra = df[INFRA_COL].mean()
            mean_target = df[TARGET_USED].mean()
            r_infra_target = np.corrcoef(
                df[INFRA_COL].fillna(0), df[TARGET_USED].fillna(0)
            )[0, 1]

            st.metric("평균 의료인프라 강도", f"{mean_infra:,.1f}")
            st.metric("평균 의료폐기물(선택 지표)", f"{mean_target:,.1f}")
            st.metric("인프라 강도 ↔ 의료폐기물 상관(r)", f"{r_infra_target:,.3f}")

        # 1) 치과병원/요양병원 → 의료인프라 강도
        st.markdown("#### (1) 치과병원·요양병원 vs 의료인프라 강도")

        infra_base = df[[INFRA_COL, DENTAL_COL, REHAB_COL]].dropna().copy()
        infra_long = infra_base.melt(
            id_vars=[INFRA_COL],
            value_vars=[DENTAL_COL, REHAB_COL],
            var_name="시설",
            value_name="value",
        )

        infra_scatter = (
            alt.Chart(infra_long)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("value:Q", title="시설 수"),
                y=alt.Y(f"{INFRA_COL}:Q", title="의료인프라 강도"),
                color=alt.Color("시설:N", title="시설 유형"),
                tooltip=["시설:N", "value:Q", alt.Tooltip(f"{INFRA_COL}:Q", title="의료인프라 강도")],
            )
            .properties(height=360)
        )
        st.altair_chart(infra_scatter, use_container_width=True)

        # 2) 의료인프라 강도 vs 의료폐기물
        st.markdown("#### (2) 의료인프라 강도 vs 의료폐기물")

        infra_waste_df = df[[INFRA_COL, TARGET_USED]].dropna()
        sc2 = (
            alt.Chart(infra_waste_df)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X(f"{INFRA_COL}:Q", title="의료인프라 강도"),
                y=alt.Y(f"{TARGET_USED}:Q", title="의료폐기물(선택 지표)"),
                tooltip=[INFRA_COL, TARGET_USED],
            )
        )
        reg2 = sc2.transform_regression(INFRA_COL, TARGET_USED, method="linear").mark_line(color="orange")
        st.altair_chart(sc2 + reg2, use_container_width=True)

        # 단순 회귀식 요약
        x = infra_waste_df[INFRA_COL].values
        y = infra_waste_df[TARGET_USED].values
        if len(x) > 1:
            b1, b0 = np.polyfit(x, y, 1)
            r = np.corrcoef(x, y)[0, 1]
            r2 = r**2

        # 3) 지도 시각화: 시도별 지리적 분포 (변수 선택)
        st.markdown("#### (3) 시도별 지리적 분포 (지도)")

        if "시도" in df.columns:
            # 시도별 집계: 인프라 강도 평균, 의료폐기물 합계, 치과/요양병원 합계
            agg_dict = {
                INFRA_COL: (INFRA_COL, "mean"),
                TARGET_USED: (TARGET_USED, "sum"),
            }
            if DENTAL_COL in df.columns:
                agg_dict[DENTAL_COL] = (DENTAL_COL, "sum")
            if REHAB_COL in df.columns:
                agg_dict[REHAB_COL] = (REHAB_COL, "sum")

            map_agg = df.groupby("시도", as_index=False).agg(**agg_dict)

            # 시도명을 지도 파일의 CTP_KOR_NM으로 매핑
            map_agg["CTP_KOR_NM"] = map_agg["시도"].map(SIDO_TO_SHP).fillna(map_agg["시도"])

            # GeoJSON 로드
            geo_data = load_geojson(GEO_FILE)

            if geo_data is not None:
                # 지도에서 선택할 수 있는 변수들
                map_var_options = {}
                if INFRA_COL in map_agg.columns:
                    map_var_options["의료인프라 강도(평균)"] = INFRA_COL
                if TARGET_USED in map_agg.columns:
                    map_var_options["의료폐기물(합계, 선택 지표)"] = TARGET_USED
                if DENTAL_COL in map_agg.columns:
                    map_var_options["치과병원 수(합계)"] = DENTAL_COL
                if REHAB_COL in map_agg.columns:
                    map_var_options["요양병원 수(합계)"] = REHAB_COL

                if not map_var_options:
                    st.info("지도에 표시할 수 있는 수치형 변수가 없습니다.")
                else:
                    sel_label_map = st.selectbox(
                        "지도에 표시할 변수 선택",
                        options=list(map_var_options.keys()),
                        index=0,
                    )
                    map_var = map_var_options[sel_label_map]

                    fig_map = px.choropleth(
                        map_agg,
                        geojson=geo_data,
                        locations="CTP_KOR_NM",
                        featureidkey="properties.CTP_KOR_NM",
                        color=map_var,
                        color_continuous_scale="Blues",
                        labels={map_var: sel_label_map},
                        hover_data={
                            "시도": True,
                            map_var: ":,.0f",
                        },
                        title=f"시도별 {sel_label_map}",
                    )
                    fig_map.update_geos(fitbounds="locations", visible=False)
                    fig_map.update_layout(height=500, margin=dict(l=0, r=0, t=60, b=0))
                    st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.info(f"'{GEO_FILE}' 파일이 없어 지도 시각화를 생략합니다.")
        else:
            st.info("시도 컬럼이 없어 지도 시각화를 그릴 수 없습니다.")

        # 4) 상관계수 요약
        r1 = np.corrcoef(df[DENTAL_COL].fillna(0), df[INFRA_COL].fillna(0))[0, 1]
        r2 = np.corrcoef(df[REHAB_COL].fillna(0), df[INFRA_COL].fillna(0))[0, 1]
        r3 = np.corrcoef(df[INFRA_COL].fillna(0), df[TARGET_USED].fillna(0))[0, 1]

        st.markdown("#### (4) 상관계수 요약 (SEM 해석용 참고치)")
        st.write(
            f"- 치과병원 ↔ 의료인프라 강도: **r = {r1:.3f}**  \n"
            f"- 요양병원 ↔ 의료인프라 강도: **r = {r2:.3f}**  \n"
            f"- 의료인프라 강도 ↔ 의료폐기물: **r = {r3:.3f}**"
        )
    else:
        st.info(
            f"'{INFRA_COL}', '{DENTAL_COL}', '{REHAB_COL}' 컬럼이 모두 있어야 인프라 탭을 그릴 수 있습니다."
        )

# -------------------------------
# 자동 인사이트 요약 (보고서용 문장 뽑기)
# -------------------------------
st.markdown("---")
st.markdown("## 🧾 자동 인사이트 요약")

insight_lines = []

# 1) 총량 기준 Top3 시도
if "시도" in df.columns:
    reg_sum = df.groupby("시도", as_index=False)[TARGET_USED].sum()
    reg_sum = reg_sum.sort_values(TARGET_USED, ascending=False)
    total_nat = reg_sum[TARGET_USED].sum()

    if len(reg_sum) >= 3:
        top3 = reg_sum.head(3)
        top3_names = ", ".join(top3["시도"].tolist())
        top3_share = top3[TARGET_USED].sum() / total_nat * 100
        insight_lines.append(
            f"- **총 의료폐기물 Top3 시도**는 {top3_names}이며, "
            f"세 지역이 전체의 약 **{top3_share:.1f}%**를 차지합니다."
        )

    # 1기관당 배출량(총 의료기관수 기준)
    facility_col = None
    for cand in ["총_의료기관수", "총_병의원수"]:
        if cand in df.columns:
            facility_col = cand
            break

    if facility_col is not None:
        reg_fac = (
            df.groupby("시도", as_index=False)[[TARGET_USED, facility_col]]
            .sum()
            .rename(columns={TARGET_USED: "waste", facility_col: "fac"})
        )
        reg_fac["waste_per_fac"] = reg_fac["waste"] / reg_fac["fac"].replace(0, np.nan)

        reg_fac = reg_fac.dropna(subset=["waste_per_fac"])
        if not reg_fac.empty:
            high = reg_fac.sort_values("waste_per_fac", ascending=False).head(1).iloc[0]
            low = reg_fac.sort_values("waste_per_fac", ascending=True).head(1).iloc[0]
            insight_lines.append(
                f"- **기관당 배출량이 가장 높은 시도**는 **{high['시도']}**로, "
                f"1기관당 평균 **{high['waste_per_fac']:.1f}** 단위의 의료폐기물을 배출합니다. "
                f"반대로 **{low['시도']}**는 기관당 배출량이 가장 낮습니다."
            )

# 2) 고령인구비율·인프라강도와의 상관관계
if "고령인구비율" in df.columns:
    r_age = np.corrcoef(df["고령인구비율"].fillna(0), df[TARGET_USED].fillna(0))[0, 1]
    direction = "높을수록 의료폐기물이 증가하는 경향" if r_age > 0 else "높을수록 의료폐기물이 감소하는 경향"
    insight_lines.append(
        f"- **고령인구비율과 의료폐기물**의 상관계수는 r ≈ {r_age:.2f}로, "
        f"고령인구 비중이 {direction}을 보입니다."
    )

if INFRA_COL in df.columns:
    r_infra = np.corrcoef(df[INFRA_COL].fillna(0), df[TARGET_USED].fillna(0))[0, 1]
    if r_infra > 0:
        infra_comment = "의료 인프라가 밀집된 지역일수록 의료폐기물도 함께 증가하는 '수요 반영형' 패턴"
    else:
        infra_comment = "인프라가 밀집된 지역에서 오히려 기관당 폐기물이 낮아지는 '효율성 효과' 패턴"
    insight_lines.append(
        f"- **의료인프라 강도와 의료폐기물**의 상관계수는 r ≈ {r_infra:.2f}로, "
        f"{infra_comment}이 나타납니다."
    )

# 3) 댓글 / 텍스트로 보여주기
if insight_lines:
    for line in insight_lines:
        st.markdown(line)
else:
    st.write("데이터에서 기본 인사이트를 추출할 수 없습니다. 컬럼 구성을 확인해주세요.")





