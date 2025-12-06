# ui_theme.py
import streamlit as st
from streamlit_option_menu import option_menu

def apply_theme() -> None:
    """밝은 종이 느낌 + 왼쪽 카드 메뉴 테마 적용"""

    # ---- 전역 CSS (배경·글자색) ----
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #f4f5fb;
            --bg-sidebar: #ffffff;
            --card-bg: #ffffff;
            --primary: #2563eb;
            --text-main: #111827;   /* ✅ 글자색 진한 검정 */
        }

        /* 전체 앱 기본 배경 & 글자색 */
        .stApp {
            background-color: var(--bg-main) !important;
            color: var(--text-main) !important;
        }

        /* 거의 모든 텍스트 요소에 검정 적용 */
        .stApp, .stApp div, .stApp span, .stApp p, .stApp li,
        .stApp label, .stApp input, .stApp textarea {
            color: var(--text-main) !important;
        }

        /* 메인 컨테이너 여백 약간 */
        [data-testid="stAppViewContainer"] {
            padding-top: 1rem;
        }

        /* 헤더(제목) 살짝 진하게 */
        h1, h2, h3, h4 {
            color: #020617 !important;
        }

        /* 사이드바 */
        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
        }
        [data-testid="stSidebar"] * {
            color: #0f172a !important;
        }

        /* metric 카드 배경/테두리 */
        [data-testid="stMetric"] {
            background-color: var(--card-bg) !important;
            border-radius: 0.75rem;
            padding: 0.75rem 0.9rem;
            border: 1px solid #e2e8f0;
        }

        /* 버튼 스타일 */
        .stButton>button {
            border-radius: 999px;
            border: none;
            background-color: var(--primary) !important;
            color: white !important;
            padding: 0.35rem 1.2rem;
            font-weight: 600;
        }
        .stButton>button:hover {
            filter: brightness(1.05);
            box-shadow: 0 8px 18px rgba(37,99,235,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- 사이드바 카드 메뉴 (option_menu) ----
    with st.sidebar:
        st.markdown("### Menu")
        selected = option_menu(
            menu_title=None,
            options=["CVRP 경로"],   # 실제 페이지는 streamlit 기본 멀티페이지로 이동하니까
            icons=["truck"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "0.5rem 0.5rem 1.2rem 0.5rem",
                    "background-color": "#f8fafc",
                    "border-radius": "1rem",
                },
                "icon": {
                    "color": "#2563eb",
                    "font-size": "20px",
                },
                "nav-link": {
                    "font-size": "16px",
                    "color": "#0f172a",         # ✅ 평상시 글자색: 검정
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#e0edff",
                },
                "nav-link-selected": {
                    "background-color": "#2563eb",
                    "color": "#ffffff",       # 선택된 메뉴는 파란 배경 + 흰 글씨
                    "font-weight": "600",
                },
            },
        )
        # 이 메뉴는 그냥 스킨용이니까 선택 값은 따로 쓰지 않아도 됨
