import streamlit as st
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="Spotify 이탈 예측 모델별 중요도", page_icon="📊", layout="wide")

# 사용자 정의 CSS
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}
.title-line {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 30px;
}
.title-green { color: #1DB954; }
.model-header {
    font-size: 28px;
    font-weight: bold;
    color: #ffffff;
    border-left: 5px solid #1DB954;
    padding-left: 15px;
    margin-top: 50px;
    margin-bottom: 20px;
}
.feature-box {
    background-color: #111111;
    border: 1px solid #333333;
    border-radius: 12px;
    padding: 15px;
    height: 140px;
    text-align: center;
    transition: transform 0.2s;
}
.feature-box:hover { border-color: #1DB954; transform: translateY(-5px); }
.feature-icon { font-size: 24px; margin-bottom: 5px; }
.feature-title { font-size: 16px; color: #1DB954; font-weight: bold; }
.feature-desc { font-size: 13px; color: #bbbbbb; }

/* --- 하단 버튼 스타일 수정 (요청 사항 반영) --- */
div[data-testid="stColumn"] div[data-testid="stButton"] > button {
    background-color: #121212 !important; /* 평소 옅은 검정색 */
    color: #ffffff !important;           /* 흰색 글자 */
    border: 1px solid #333333 !important;
    border-radius: 25px;                 /* 둥근 타원형 스타일 */
    width: 100%;
    height: 50px;
    font-size: 16px;
    font-weight: 500;
    transition: all 0.3s ease;
}

/* 마우스를 올렸을 때와 클릭할 때 (Hover & Focus) */
div[data-testid="stColumn"] div[data-testid="stButton"] > button:hover,
div[data-testid="stColumn"] div[data-testid="stButton"] > button:active,
div[data-testid="stColumn"] div[data-testid="stButton"] > button:focus {
    background-color: #1DB954 !important; /* 스포티파이 녹색 */
    color: #000000 !important;           /* 글자는 검정색으로 */
    border: 1px solid #1DB954 !important;
}

.footer { text-align: center; font-size: 14px; color: #888888; margin-top: 60px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-line">모델별 <span class="title-green">특성 중요도(Feature Importance)</span> 분석</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. 머신러닝 (Random Forest) 섹션
# ---------------------------------------------------------
st.markdown('<div class="model-header">1. 머신러닝 (Random Forest) 주요 지표</div>', unsafe_allow_html=True)

rf_labels = ["subscription_type", "offline_listening", "ads_listened_per_week", "country", "satisfaction_score", "songs_played_per_day"]
rf_values = [53.07, 24.23, 19.99, 0.61, 0.52, 0.24]
rf_icons = ["🎫", "🎧", "📻", "🌎", "😊", "🎶"]
rf_descs = ["가장 핵심적인 이탈 요인", "오프라인 활용도", "광고 노출 영향", "국가별 환경", "서비스 만족도", "하루 재생 빈도"]

cols_rf = st.columns(6)
for col, label, icon, desc in zip(cols_rf, rf_labels, rf_icons, rf_descs):
    with col:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{label}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

fig_rf = go.Figure(go.Bar(
    x=rf_values[::-1], y=rf_labels[::-1], orientation='h',
    marker_color='#1DB954', text=[f"{v}%" for v in rf_values[::-1]], textposition='outside'
))
fig_rf.update_layout(
    title="Random Forest 특성 중요도 (%)",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
    xaxis=dict(showgrid=True, gridcolor='#333333', range=[0, 60]), height=400
)
st.plotly_chart(fig_rf, use_container_width=True)

# ---------------------------------------------------------
# 2. 딥러닝 (DNN) 섹션
# ---------------------------------------------------------
st.markdown('<div class="model-header">2. 딥러닝 (DNN) 주요 지표</div>', unsafe_allow_html=True)

dnn_labels = ["subscription_type", "offline_listening", "ads_listened_per_week", "listening_time", "songs_played_per_day", "ad_burden"]
dnn_values = [49.25, 26.04, 19.36, 1.37, 0.94, 0.72]
dnn_icons = ["🎫", "🎧", "📻", "⏳", "🎶", "⚠️"]
dnn_descs = ["구독 유형의 높은 기여도", "오프라인 재생 비중", "주간 광고 청취", "총 청취 시간", "일별 곡 재생수", "광고 체감 부담"]

cols_dnn = st.columns(6)
for col, label, icon, desc in zip(cols_dnn, dnn_labels, dnn_icons, dnn_descs):
    with col:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{label}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

fig_dnn = go.Figure(go.Bar(
    x=dnn_values[::-1], y=dnn_labels[::-1], orientation='h',
    marker_color='#3498db', text=[f"{v}%" for v in dnn_values[::-1]], textposition='outside'
))
fig_dnn.update_layout(
    title="DNN 특성 중요도 (%)",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
    xaxis=dict(showgrid=True, gridcolor='#333333', range=[0, 60]), height=400
)
st.plotly_chart(fig_dnn, use_container_width=True)

# ---------------------------------------------------------
# 네비게이션 및 푸터 (요청 스타일 적용)
# ---------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
nav_cols = st.columns([1.5, 7, 1.5]) # 양 끝 버튼 배치를 위해 비율 조정
with nav_cols[0]:
    if st.button("🏠 Home"): st.switch_page("Home.py")
with nav_cols[2]:
    if st.button("Next ➡"): st.switch_page("pages/model_comparison.py")

st.markdown("---")
st.markdown('<div class="footer">Spotify Churn Prediction Project<br>RF vs DNN Feature Importance Comparison</div>', unsafe_allow_html=True)