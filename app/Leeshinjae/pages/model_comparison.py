import streamlit as st
import json
import os

# 페이지 설정
st.set_page_config(page_title="모델 성능 비교", page_icon="📊", layout="wide")

# 1. 데이터 로드 함수 (JSON 연동)
def load_metrics():
    metrics_path = os.path.join("..", "JangWansik", "03_trained_model", "model_metrics.json")
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        pass
    return {}

metrics_data = load_metrics()

# 2. 사용자 정의 CSS
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}
.title-white {
    font-size: 48px; color: #ffffff; font-weight: bold; margin-left: 20px; line-height: 1.2;
}
.title-green {
    font-size: 48px; color: #1DB954; font-weight: bold; margin-left: 20px; margin-bottom: 30px; line-height: 1.2;
}
/* 핵심 지표 소형 박스 */
.small-box {
    background-color: #111111;
    border: 1px solid #1DB954;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 200px;
}
.small-title {
    font-size: 16px; color: #1DB954; font-weight: bold; margin-bottom: 8px;
}
.small-value {
    font-size: 28px; color: #ffffff !important; font-weight: 800;
}
/* 모델 상세 대형 박스 */
.large-box {
    background-color: #1c1c1c;
    border: 1px solid #1DB954;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 10px;
    height: 200px;
}
.large-icon { font-size: 30px; margin-bottom: 10px; }
.large-title { font-size: 18px; color: #1DB954; font-weight: bold; margin-bottom: 8px; }
.large-desc { font-size: 13px; color: #bbbbbb; margin-bottom: 12px; min-height: 32px; }
.large-score { font-size: 20px; color: #ffffff !important; font-weight: bold; }

/* 하단 네비게이션 버튼 스타일 (기존 디자인 유지) */
div[data-testid="stColumn"] div[data-testid="stButton"] > button {
    background-color: #111111;
    border: 1px solid #1DB954;
    border-radius: 12px;
    color: white;
    width: 100%;
    height: 60px;
    font-size: 18px;
    font-weight: bold;
    transition: all 0.3s ease;
    margin-top: 20px;
}

div[data-testid="stColumn"] div[data-testid="stButton"] > button:hover {
    background-color: #1DB954;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# 3. 데이터 추출
rf_metrics = metrics_data.get("RandomForest", {})
dl_metrics = metrics_data.get("Deep Learning (DNN)", {})

# 4. 메인 콘텐츠 레이아웃
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown('<div class="title-white">다양한 모델</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-green">성능 비교</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 핵심 성능 지표 (Avg)")
    
    avg_acc = (rf_metrics.get("Accuracy", 0.8115) + dl_metrics.get("Accuracy", 0.8131)) / 2 * 100
    avg_f1 = (rf_metrics.get("F1-Score", 0.744) + dl_metrics.get("F1-Score", 0.745)) / 2
    
    small_cols = st.columns(3)
    summary_metrics = [
        {"title": "Avg Accuracy", "value": f"{avg_acc:.2f}%"},
        {"title": "Best F1-Score", "value": f"{max(rf_metrics.get('F1-Score', 0), dl_metrics.get('F1-Score', 0)):.3f}"},
        {"title": "Threshold", "value": f"{rf_metrics.get('Best Threshold', 0.5)*100:.0f}%"}
    ]
    
    for col, m in zip(small_cols, summary_metrics):
        with col:
            st.markdown(f"""
            <div class="small-box">
                <div class="small-title">{m['title']}</div>
                <div class="small-value">{m['value']}</div>
            </div>
            """, unsafe_allow_html=True)

with right_col:
    model_list = [
        {"icon": "🌲", "title": "Random Forest", "desc": "다수의 결정 트리로부터 분류", "score": f"{rf_metrics.get('Accuracy', 0.812)*100:.1f}%"},
        {"icon": "🧠", "title": "Deep Learning", "desc": "TensorFlow 기반 DNN 모델", "score": f"{dl_metrics.get('Accuracy', 0.813)*100:.1f}%"},
        {"icon": "⚡", "title": "XGBoost", "desc": "성능 최적화 부스팅 알고리즘", "score": "80.5%"},
        {"icon": "📈", "title": "LSTM", "desc": "시계열 데이터 패턴 학습", "score": "79.8%"}
    ]
    
    for i in range(0, len(model_list), 2):
        row = st.columns(2)
        for col, model in zip(row, model_list[i:i+2]):
            with col:
                st.markdown(f"""
                <div class="large-box">
                    <div class="large-icon">{model['icon']}</div>
                    <div class="large-title">{model['title']}</div>
                    <div class="large-desc">{model['desc']}</div>
                    <div class="large-score">정확도: {model['score']}</div>
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. 하단 네비게이션 버튼 (양 끝 정렬)
st.markdown("<br>", unsafe_allow_html=True)
nav_cols = st.columns(15)

with nav_cols[0]: # 좌측 끝 (제목 컬럼 라인)
    if st.button("🏠 Home"):
        st.switch_page("Home.py")

with nav_cols[14]: # 우측 끝 (모델 카드 우측 끝 라인)
    if st.button("Next ➡️"):
        st.switch_page("pages/ChurnCheck.py") # 다음 예측 페이지로 이동

# ---------------------------------------------------------
# 푸터
st.markdown("---")
st.caption("© 2025 Spotify Churn Prediction Project - Data synchronized with model_metrics.json")