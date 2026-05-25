import streamlit as st

# 1. 페이지 설정
st.set_page_config(
    page_title="Spotify 고객 이탈 예측",
    page_icon="🎧",
    layout="wide"
)

# 2. 사용자 정의 CSS
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}

/* 텍스트 스타일 */
.title {
    text-align: center;
    font-size: 26px;
    color: #ffffff;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 6px;
}
.headline-white {
    text-align: center;
    font-size: 96px;
    color: #ffffff;
    font-weight: 800;
    margin: 0;
    line-height: 1.2;
}
.headline-green {
    text-align: center;
    font-size: 96px;
    color: #1DB954;
    font-weight: 800;
    margin: 2px 0 12px 0;
    line-height: 1.2;
}

.description {
    color: #ffffff;
    text-align: center;
    font-size: 16px;
    margin: 30px auto 40px auto;
    max-width: 900px;
    line-height: 1.6;
}

/* 메트릭 버튼 스타일 수정 */
div[data-testid="stButton"] > button {
    background-color: #111111;
    border: 1px solid #1DB954;
    border-radius: 12px;
    width: 150%;
    height: 160px; /* 고정 높이로 동일한 크기 유지 */
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    color: white;
    cursor: pointer;
    padding: 20px;
    white-space: pre-line; /* 줄바꿈(\n) 인식 */
    transition: all 0.3s ease;
    line-height: 1.4;
}

/* 버튼 호버 효과 */
div[data-testid="stButton"] > button:hover {
    background-color: #1DB954;
    color: black;
    border-color: #1DB954;
}

/* 버튼 내 텍스트 스타일 강제 적용 (Streamlit 기본 스타일 덮어쓰기) */
div[data-testid="stButton"] > button p {
    font-size: 20px !important; /* 설명 글자 크기 */
    font-weight: 500;
}

/* 첫 번째 줄(Title)만 크게 만들기 위한 트릭 (선택 사항) */
/* 만약 타이틀만 따로 크게 하고 싶다면 아래와 같이 텍스트 구성을 조정합니다. */
</style>
""", unsafe_allow_html=True)

# 3. 메인 화면 구성
st.markdown('<div class="title">🎧 Spotify Customer Analytics 🎵</div>', unsafe_allow_html=True)
st.markdown('<div class="headline-white">가입 고객</div>', unsafe_allow_html=True)
st.markdown('<div class="headline-green">이탈 예측</div>', unsafe_allow_html=True)

st.markdown("""
<div class="description">
머신러닝과 딥러닝을 활용한 Spotify 고객 이탈 예측 모델 구축 및 배포 프로젝트입니다.<br>
고객 행동 데이터를 기반으로 이탈 가능성을 실시간으로 예측하여 비즈니스 전략 수립에 도움을 줍니다.
</div>
""", unsafe_allow_html=True)

# 메트릭 박스 구성
cols = st.columns(5)
metrics = [
    {"title": "4", "desc": "Pipeline step", "page": "pages/pipeline.py"},
    {"title": "6", "desc": "Key  Features", "page": "pages/Key_features.py"},
    {"title": "ML/DL", "desc": "예측모델  설정", "page": "pages/model_comparison.py"},
    {"title": "+- 82%", "desc": "이탈  예측하기", "page": "pages/ChurnCheck.py"},
    {"title": "Real-time", "desc": "이탈  대응단계", "page": "pages/business_strategy.py"}
]

for col, m in zip(cols, metrics):
    with col:
        # HTML 대신 줄바꿈(\n)을 사용하여 텍스트 전달
        # 타이틀을 강조하고 싶을 경우 이모지 등을 섞어 시각적 구분 가능
        button_text = f"{m['title']}\n\n\n\n\n\n{m['desc']}"
        clicked = st.button(button_text, key=f"btn_{m['title']}")
        
        if clicked:
            st.switch_page(m["page"])

# 푸터
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.caption("© 2025 Spotify Churn Prediction Project")