# ui_theme.py
import streamlit as st

def apply_theme(name: str = "neo-dark"):
    """name 값에 따라 다른 스킨 적용"""
    if name == "neo-dark":
        _neo_dark()
    elif name == "paper-light":
        _paper_light()
    elif name == "glass-dark":
        _glass_dark()
    else:
        _neo_dark()  # 기본값


def _neo_dark():
    st.markdown(
        """
        <style>
        /* 전체 배경 그라데이션 */
        body {
            background: radial-gradient(circle at top left, #1f2937 0, #020617 40%, #000000 100%);
        }

        .main .block-container {
            max-width: 1200px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* 타이틀 그라데이션 */
        h1 {
            font-size: 2.6rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.03em;
            background: linear-gradient(90deg, #f97316, #facc15, #4ade80);
            -webkit-background-clip: text;
            color: transparent;
        }

        /* 사이드바 */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617, #020617 40%, #111827 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.4);
        }

        /* metric 카드 */
        [data-testid="metric-container"] {
            background: radial-gradient(circle at top left, #111827, #020617);
            border-radius: 18px;
            padding: 1rem 1.3rem;
            border: 1px solid rgba(148, 163, 184, 0.5);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.9);
        }
        [data-testid="metric-container"] > div {
            color: #e5e7eb !important;
        }

        /* 버튼 */
        .stButton > button {
            border-radius: 999px;
            padding: 0.4rem 1.2rem;
            font-weight: 600;
            border: 1px solid #f97316;
            background: linear-gradient(90deg, #f97316, #fb923c);
            color: #020617;
        }
        .stButton > button:hover {
            filter: brightness(1.1);
            box-shadow: 0 0 25px rgba(248, 113, 113, 0.4);
        }

        /* expander / 데이터프레임 카드 */
        details {
            border-radius: 18px !important;
            background: rgba(15, 23, 42, 0.9) !important;
            border: 1px solid rgba(148, 163, 184, 0.4) !important;
            backdrop-filter: blur(12px);
        }

        .stDataFrame, .stTable {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.4);
            box-shadow: 0 15px 30px rgba(15, 23, 42, 0.8);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _paper_light():
    st.markdown(
        """
        <style>
        body {
            background: #f3f4f6;
        }
        .main .block-container {
            max-width: 1100px;
            padding-top: 1.5rem;
        }
        h1 {
            font-size: 2.4rem !important;
            font-weight: 800 !important;
            color: #111827 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e5e7eb;
        }
        [data-testid="metric-container"] {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 0.9rem 1.1rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
        }
        .stButton > button {
            border-radius: 8px;
            padding: 0.4rem 1rem;
            border: 1px solid #2563eb;
            background-color: #3b82f6;
            color: white;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _glass_dark():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(135deg, #000000, #020617, #0f172a);
        }
        .main .block-container {
            max-width: 1200px;
            padding-top: 2rem;
        }
        h1 {
            font-size: 2.7rem !important;
            font-weight: 900 !important;
            background: linear-gradient(120deg, #38bdf8, #a855f7, #ec4899);
            -webkit-background-clip: text;
            color: transparent;
        }
        [data-testid="stSidebar"] {
            background: transparent;
            border-right: 1px solid rgba(148, 163, 184, 0.3);
            backdrop-filter: blur(22px);
        }
        [data-testid="metric-container"] {
            background: rgba(15, 23, 42, 0.75);
            border-radius: 20px;
            padding: 1rem 1.2rem;
            border: 1px solid rgba(148, 163, 184, 0.6);
            backdrop-filter: blur(16px);
        }
        .stButton > button {
            border-radius: 999px;
            padding: 0.4rem 1.4rem;
            background: linear-gradient(120deg, #38bdf8, #a855f7);
            border: none;
            color: #0b1120;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
