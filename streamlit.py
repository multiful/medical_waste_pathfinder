# app.py — 의료폐기물 분석 대시보드
# 데이터: final_df.csv (시도별 의료폐기물 + 병원/의원 수 + 인구/인프라 등)

from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import streamlit as st

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
st.caption("데이터: final_df.csv (시도×연도 단위 의료폐기물 및 의료 인프라 지표)")

DATA_FILE = "final_df.csv"

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
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 문자열 공백 정리
    if "시도" in df.columns:
        df["시도"] = df["시도"].astype(str).str.strip()
    return df

# -------------------------------
# 데이터 로딩
# -------------------------------
if not Path(DATA_FILE).exists():
    st.error(f"'{DATA_FILE}' 파일을 찾을 수 없습니다. 같은 폴더에 final_df.csv를 두고 다시 실행해 주세요.")
    st.stop()

df_raw = load_data(DATA_FILE)

# 주요 컬럼 이름들 (없는 경우도 대비)
TARGET_COL = "지역별_의료폐기물"
TARGET_TRANS_COL = "지역별_의료폐기물_TRANS"  # 있으면 선택해서 사용
DENTAL_COL = "치과병원"
REHAB_COL = "요양병원"
INFRA_COL = "의료인프라_강도"

FACILITY_HOSP_COLS = [
    "상급종합병원", "종합병원", "치과병원",
    "한방병원", "요양병원", "정신병원",
]
FACILITY_CLINIC_COLS = ["의원", "치과의원", "한의원"]

num_cols_all = df_raw.select_dtypes(include=[np.number]).columns.tolist()

# -------------------------------
# 사이드바 필터
# -------------------------------
with st.sidebar:
    st.header("필터")

    # 연도 필터
    df = df_raw.copy()
    if "연도" in df.columns:
        years = sorted(df["연도"].dropna().unique().tolist())
        sel_years = st.multiselect(
            "연도 선택",
            options=years,
            default=years,
        )
        if sel_years:
            df = df[df["연도"].isin(sel_years)]
        st.caption(f"선택된 연도: {', '.join(map(str, sel_years)) if sel_years else '전체'}")
    else:
        st.info("연도 컬럼이 없어 연도 필터는 표시하지 않습니다.")

    # 시도 필터
    if "시도" in df.columns:
        sidos = sorted(df["시도"].dropna().unique().tolist())
        sel_sidos = st.multiselect(
            "시도 선택",
            options=sidos,
            default=sidos,
        )
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

target_series = df[TARGET_COL] if TARGET_COL in df.columns else df[TARGET_USED]
total_waste = target_series.sum()
mean_waste_per_region = df.groupby("시도")[TARGET_COL].sum().mean() if "시도" in df.columns and TARGET_COL in df.columns else np.nan

if "치과병원" in df.columns:
    total_dental = df[DENTAL_COL].sum()
    waste_per_dental = total_waste / total_dental if total_dental > 0 else np.nan
else:
    waste_per_dental = np.nan

if "시도" in df.columns:
    top_region = (
        df.groupby("시도")[TARGET_COL]
        .sum()
        .sort_values(ascending=False)
        .head(1)
    )
    top_region_name = top_region.index[0]
    top_region_val = int(top_region.iloc[0])
else:
    top_region_name, top_region_val = "-", np.nan

with k1:
    st.metric("총 의료폐기물 배출량", f"{int(total_waste):,} 톤")
with k2:
    if not np.isnan(mean_waste_per_region):
        st.metric("시도별 평균 의료폐기물", f"{int(mean_waste_per_region):,} 톤")
    else:
        st.metric("시도별 평균 의료폐기물", "N/A")
with k3:
    if not np.isnan(waste_per_dental):
        st.metric("치과병원 1기관당 평균 의료폐기물", f"{waste_per_dental:,.1f} 톤")
    else:
        st.metric("치과병원 1기관당 평균 의료폐기물", "N/A")
with k4:
    st.metric("의료폐기물 최다 배출 시도", f"{top_region_name} ({top_region_val:,} 톤)" if not np.isnan(top_region_val) else "N/A")

st.markdown("---")

# -------------------------------
# 탭 레이아웃
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["시도별 비교", "시설유형별 병원당 폐기물", "상관·회귀 분석", "의료 인프라(SEM 관점)"]
)

# -------------------------------
# Tab1: 시도별 의료폐기물 & 치과병원당 폐기물
# -------------------------------
with tab1:
    st.markdown("### 시도별 의료폐기물 및 치과병원당 폐기물")

    if {"시도", TARGET_COL}.issubset(df.columns):
        grouped = df.groupby("시도", as_index=False).agg(
            의료폐기물=(TARGET_COL, "sum"),
            치과병원=(DENTAL_COL, "sum") if DENTAL_COL in df.columns else ("시도", "size"),
        )
        if DENTAL_COL in df.columns:
            grouped["치과병원_당_폐기물"] = grouped["의료폐기물"] / grouped["치과병원"].replace(0, np.nan)

        c1, c2 = st.columns([2, 1], gap="large")

        with c1:
            base = grouped.sort_values("치과병원_당_폐기물" if "치과병원_당_폐기물" in grouped.columns else "의료폐기물")
            bar = (
                alt.Chart(base)
                .mark_bar()
                .encode(
                    x=alt.X("시도:N", sort=None),
                    y=alt.Y(
                        "치과병원_당_폐기물:Q",
                        title="치과병원 1기관당 의료폐기물(톤)",
                    )
                    if "치과병원_당_폐기물" in base.columns
                    else alt.Y("의료폐기물:Q", title="의료폐기물(톤)"),
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
                    y=alt.Y("값:Q", title="값(톤 / 기관수)"),
                    color="지표:N",
                    tooltip=["시도:N", "지표:N", "값:Q"],
                )
                .properties(height=380)
            )
            st.altair_chart(line, use_container_width=True)

        with st.expander("표 보기 (시도별 집계)"):
            st.dataframe(grouped.sort_values("의료폐기물", ascending=False), use_container_width=True)
    else:
        st.warning("시도 또는 지역별_의료폐기물 컬럼이 없어 시도별 비교를 그릴 수 없습니다.")

# -------------------------------
# Tab2: 시설유형별 병원당 의료폐기물
# -------------------------------
with tab2:
    st.markdown("### 전국 시설유형별 병원당 의료폐기물")

    if TARGET_COL in df.columns:
        total_waste_all = df[TARGET_COL].sum()

        facility_totals = []
        for col in FACILITY_HOSP_COLS:
            if col in df.columns:
                tot = df[col].sum()
                if tot > 0:
                    facility_totals.append(
                        {"facility": col, "병원수": tot, "waste_per_facility": total_waste_all / tot}
                    )

        if facility_totals:
            fac_df = pd.DataFrame(facility_totals)

            c1, c2 = st.columns([2, 1], gap="large")
            with c1:
                bar = (
                    alt.Chart(fac_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("facility:N", title="시설 유형"),
                        y=alt.Y("waste_per_facility:Q", title="병원 1기관당 의료폐기물(톤)"),
                        tooltip=["facility", "병원수", "waste_per_facility"],
                        color=alt.Color("waste_per_facility:Q", scale=alt.Scale(scheme="reds")),
                    )
                    .properties(height=380)
                )
                st.altair_chart(bar, use_container_width=True)

            with c2:
                pie = px.pie(
                    fac_df,
                    values="waste_per_facility",
                    names="facility",
                    title="시설유형별 병원당 폐기물 비중",
                    hole=0.4,
                )
                pie.update_traces(textinfo="percent+label")
                st.plotly_chart(pie, use_container_width=True)

            with st.expander("표 보기 (시설유형별 병원당 의료폐기물)"):
                st.dataframe(fac_df.sort_values("waste_per_facility", ascending=False), use_container_width=True)
        else:
            st.warning("병원 계열 시설 컬럼(상급종합병원, 종합병원, 치과병원, 한방병원, 요양병원 등)을 찾지 못했습니다.")
    else:
        st.warning("지역별_의료폐기물 컬럼이 없어 시설유형 분석을 수행할 수 없습니다.")

# -------------------------------
# Tab3: 상관·회귀 분석
# -------------------------------
with tab3:
    st.markdown("### 의료폐기물과 의료 인프라 지표 간 상관·회귀 분석")

    if TARGET_COL not in df.columns:
        st.warning("지역별_의료폐기물 컬럼이 없어 상관 분석을 수행할 수 없습니다.")
    else:
        # 상관계수 (Pearson)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()[TARGET_COL].drop(labels=[TARGET_COL])
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

        with st.expander("표 보기 (상관계수)"):
            st.dataframe(corr_df.drop(columns="abs_r").sort_values("Pearson r", ascending=False), use_container_width=True)

        st.markdown("---")
        st.markdown("#### 특정 시설 수 vs 의료폐기물 (산점도 + 회귀선)")

        # 산점도에서 x축에 쓸 후보(의료기관 수 관련 변수)
        candidate_xcols = [c for c in FACILITY_HOSP_COLS + FACILITY_CLINIC_COLS if c in df.columns]
        if not candidate_xcols:
            candidate_xcols = [c for c in numeric_cols if c != TARGET_COL]

        sel_x = st.selectbox("x축 변수 선택", options=candidate_xcols, index=0)

        scatter_df = df[[sel_x, TARGET_COL]].dropna()

        sc = (
            alt.Chart(scatter_df)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X(f"{sel_x}:Q", title=sel_x),
                y=alt.Y(f"{TARGET_COL}:Q", title="지역별 의료폐기물"),
                tooltip=[sel_x, TARGET_COL],
            )
        )

        reg = (
            sc.transform_regression(sel_x, TARGET_COL, method="linear")
            .mark_line(color="orange")
        )

        st.altair_chart(sc + reg, use_container_width=True)
        st.caption("※ 점 하나는 (시도×연도) 또는 분석 단위 하나를 의미. 직선 기울기는 단순 선형회귀 계수에 해당.")

# -------------------------------
# Tab4: 의료 인프라(SEM 관점)
# -------------------------------
with tab4:
    st.markdown("### 의료 인프라 강도와 의료폐기물 (SEM 구조 해석용)")

    if {INFRA_COL, DENTAL_COL, REHAB_COL}.issubset(df.columns) and TARGET_USED in df.columns:
        info_col1, info_col2 = st.columns([2, 1])

        with info_col1:
            st.markdown(
                """
**가설(H4)**  
- 치과병원·요양병원 증가 → 의료인프라 강도(인구 대비 병의원 수) 증가  
- 의료인프라 강도 증가 → 의료폐기물 증가  

이 탭은 위 SEM 구조를 이해하기 위한 기초 EDA를 보여줍니다.
                """
            )

        with info_col2:
            st.write("사용 컬럼")
            st.code(
                f"""
의료폐기물: {TARGET_USED}
의료인프라 강도: {INFRA_COL}
치과병원 수: {DENTAL_COL}
요양병원 수: {REHAB_COL}
""",
                language="text",
            )

        # 1) 치과병원/요양병원 → 의료인프라 강도
        st.markdown("#### (1) 치과병원·요양병원 vs 의료인프라 강도")

        infra_df = df[[DENTAL_COL, REHAB_COL, INFRA_COL]].dropna()

        infra_scatter = (
            alt.Chart(infra_df)
            .transform_fold(
                [DENTAL_COL, REHAB_COL],
                as_=["시설", "value"],
            )
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("value:Q", title="시설 수"),
                y=alt.Y(f"{INFRA_COL}:Q", title="의료인프라 강도"),
                color=alt.Color("시설:N", title="시설 유형"),
                tooltip=["시설", "value", INFRA_COL],
            )
            .properties(height=360)
        )
        st.altair_chart(infra_scatter, use_container_width=True)

        # 2) 의료인프라 강도 → 의료폐기물
        st.markdown("#### (2) 의료인프라 강도 vs 의료폐기물")

        infra_waste_df = df[[INFRA_COL, TARGET_USED]].dropna()
        sc2 = (
            alt.Chart(infra_waste_df)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X(f"{INFRA_COL}:Q", title="의료인프라 강도"),
                y=alt.Y(f"{TARGET_USED}:Q", title="의료폐기물"),
                tooltip=[INFRA_COL, TARGET_USED],
            )
        )
        reg2 = (
            sc2.transform_regression(INFRA_COL, TARGET_USED, method="linear")
            .mark_line(color="orange")
        )
        st.altair_chart(sc2 + reg2, use_container_width=True)

        # 간단한 상관 요약
        r1 = np.corrcoef(df[DENTAL_COL].fillna(0), df[INFRA_COL].fillna(0))[0, 1]
        r2 = np.corrcoef(df[REHAB_COL].fillna(0), df[INFRA_COL].fillna(0))[0, 1]
        r3 = np.corrcoef(df[INFRA_COL].fillna(0), df[TARGET_USED].fillna(0))[0, 1]

        st.markdown("#### (3) 상관계수 요약 (SEM 해석용 참고치)")
        st.write(
            f"- 치과병원 ↔ 의료인프라 강도: **r = {r1:.3f}**  \n"
            f"- 요양병원 ↔ 의료인프라 강도: **r = {r2:.3f}**  \n"
            f"- 의료인프라 강도 ↔ 의료폐기물: **r = {r3:.3f}**"
        )
        st.caption("※ 실제 SEM 결과(직접/간접효과·적합도)는 논문/보고서에서 별도로 제시하고, 이 대시보드는 그 기초가 되는 관계를 시각화하는 용도.")
    else:
        st.info(
            f"'{INFRA_COL}', '{DENTAL_COL}', '{REHAB_COL}' 컬럼이 모두 있어야 인프라 탭을 그릴 수 있습니다."
        )
