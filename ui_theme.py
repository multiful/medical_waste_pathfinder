# ui_theme.py
# 밝은 톤의 대시보드 공통 스킨

import streamlit as st


def apply_theme() -> None:
    """Streamlit 전역 UI 스킨 적용 (라이트 테마)."""
    st.markdown(
        """
        <style>
        /* 전체 배경 & 기본 텍스트 */
        .stApp {
            background: #f5f7fb;
            color: #111827;
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px;
        }

        /* 사이드바 카드 느낌 */
        [data-testid="stSidebar"] > div {
            background-color: #ffffff !important;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
            border-right: 1px solid #e5e7eb;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2 {
            color: #111827 !important;
        }
        [data-testid="stSidebar"] {
            color: #374151 !important;
        }

        /* Metric 카드 */
        .stMetric {
            background-color: #ffffff !important;
            border-radius: 0.8rem !important;
            padding: 1rem 1.2rem !important;
            box-shadow: 0 12px 30px rgba(148, 163, 184, 0.25) !important;
            border: 1px solid #e5e7eb !important;
        }
        .stMetric label {
            color: #6b7280 !important;
            font-size: 0.78rem !important;
            text-transform: uppercase;
            letter-spacing: 0.09em;
        }
        .stMetric div[data-testid="stMetricValue"] {
            color: #111827 !important;
        }

        /* Expander */
        div[data-testid="stExpander"] {
            background-color: #ffffff !important;
            border-radius: 0.9rem !important;
            border: 1px solid #e5e7eb !important;
        }
        div[data-testid="stExpander"] > details > summary {
            color: #111827 !important;
            font-weight: 600 !important;
        }

        /* 데이터프레임 헤더 */
        .stDataFrame thead tr th {
            background-color: #f9fafb !important;
            color: #374151 !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
