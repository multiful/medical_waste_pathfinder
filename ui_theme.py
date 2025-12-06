# ui_theme.py
import streamlit as st


def apply_theme(theme_name: str = "paper-light") -> None:
    """
    - config.toml 에서 base/light + primaryColor 세팅한 걸 기본으로 쓰고
    - 여기서는 글자색 / 사이드바 / 멀티셀렉트 pill 스타일만 살짝 손본다.
    """
    css = """
    <style>
    /* 전체 텍스트는 진한 회색 계열로 통일 */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #111827 !important;
    }

    /* 헤더 배경 투명하게 (위에 회색 띠 안 생기게) */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* 메인 영역 살짝 여백 */
    section.main > div {
        padding-top: 0.5rem;
    }

    /* 사이드바는 흰색 카드 느낌으로 */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border-right: 1px solid #E5E7EB !important;
    }

    /* 사이드바 안의 제목(label) 색 진하게 */
    [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] > label {
        color: #111827 !important;
        font-weight: 600;
    }

    /* 멀티셀렉트/셀렉트 박스 컨테이너 (검정 배경 제거) */
    div[data-baseweb="select"] > div {
        background-color: #F9FAFB !important;
        border-radius: 12px !important;
        border: 1px solid #E5E7EB !important;
    }

    /* 멀티셀렉트 선택된 값 pill → 파란색 뱃지로 통일 */
    div[data-baseweb="tag"] {
        background-color: #2563EB !important;  /* 파란색 */
        color: #FFFFFF !important;
        border-radius: 999px !important;
        border: none !important;
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }
    div[data-baseweb="tag"] span {
        color: #FFFFFF !important;
    }
    div[data-baseweb="tag"] svg {
        fill: #FFFFFF !important;
    }

    /* 버튼 기본도 파란색 느낌으로 정리 (있으면) */
    button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-radius: 999px !important;
        border: none !important;
    }
    button[kind="secondary"] {
        color: #2563EB !important;
        border-color: #2563EB !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
