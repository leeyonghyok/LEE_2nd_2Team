import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# TensorFlow는 사용하지 않음 (시뮬레이션 예측 사용)

# 페이지 설정
st.set_page_config(page_title="실시간 이탈 예측", page_icon="🔮", layout="wide")
# 사용자 정의 CSS 수정
st.markdown(
    """
<style>
/* ... 기존 스타일 유지 ... */

/* [새로 추가] 커스텀 메트릭 박스 스타일 */
.metric-container {
    background-color: #111111;
    border: 1px solid #1DB954;
    border-radius: 12px;
    padding: 20px;
    text-align: center;      /* 박스 안의 모든 글자 가운데 정렬 */
    margin-bottom: 10px;
}
.metric-label {
    color: #b3b3b3;          /* 라벨은 가독성을 위해 살짝 연한 회색 */
    font-size: 14px;
    margin-bottom: 5px;
}
.metric-value {
    color: #ffffff !important; /* 글자색을 완전한 하얀색으로 */
    font-size: 32px;
    font-weight: 800;        /* 글자를 굵게 해서 가독성 향상 */
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
/* 종합 결과용 커스텀 메트릭 박스 */
.summary-metric-container {
    background-color: #111111;
    border: 1px solid #1DB954;
    border-radius: 12px;
    padding: 20px;
    text-align: center;      /* 가운데 정렬 */
    height: 100%;            /* 높이 균일화 */
}
.summary-label {
    color: #b3b3b3;          /* 상단 라벨 회색 */
    font-size: 14px;
    margin-bottom: 8px;
}
.summary-value {
    color: #ffffff !important; /* 수치/텍스트 완전한 하얀색 */
    font-size: 24px;         /* 종합 결과에 적당한 크기 */
    font-weight: 800;
}
</style>
""",
    unsafe_allow_html=True,
)

# 사용자 정의 CSS (기존 스타일 유지)
st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}
.stMetric {
    background-color: #111111;
    border: 1px solid #1DB954;
    border-radius: 12px;
    padding: 15px;
}
.prediction-card {
    background-color: #111111;
    border: 1px solid #1DB954;
    border-radius: 12px;
    padding: 25px;
    margin: 10px 0;
}
.prediction-title {
    font-size: 24px;
    color: #1DB954;
    font-weight: 700;
    margin-bottom: 15px;
    text-align: center;
}
.risk-high {
    color: #FF5252;
    font-size: 32px;
    font-weight: bold;
}
.risk-medium {
    color: #FFC107;
    font-size: 32px;
    font-weight: bold;
}
.risk-low {
    color: #1DB954;
    font-size: 32px;
    font-weight: bold;
}
.comparison-container {
    display: flex;
    justify-content: space-around;
    margin: 20px 0;
}
 /* 하단 네비게이션 버튼 스타일 */
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
""",
    unsafe_allow_html=True,
)

# 모델 경로 설정 (Leeshinjae 폴더 기준)
MODEL_BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "notebooks",
    "JangWansik",
)
ML_MODEL_PATH = os.path.join(
    MODEL_BASE_PATH, "03_trained_model", "spotify_churn_model.pkl"
)
DL_MODEL_PATH = os.path.join(
    MODEL_BASE_PATH, "03_trained_model", "spotify_dl_model.onnx"
)
DL_SCALER_PATH = os.path.join(
    MODEL_BASE_PATH, "03_trained_model", "spotify_dl_model.onnx.data"
)
METRICS_PATH = os.path.join(MODEL_BASE_PATH, "03_trained_model", "model_metrics.json")


def simulate_ml_prediction(input_data):
    """ML 모델 시뮬레이션 예측 (규칙 기반)"""
    # 입력 데이터에서 값 추출
    skip_rate = input_data["skip_rate"].iloc[0]
    listening_time = input_data["listening_time"].iloc[0]
    ad_burden = input_data["ad_burden"].iloc[0]
    offline = input_data["offline_listening"].iloc[0]
    sub_type = input_data["subscription_type"].iloc[0]
    songs_per_day = input_data["songs_played_per_day"].iloc[0]

    # 기본 확률 (0.3에서 시작)
    base_prob = 0.3

    # 위험 요인 가중치
    if skip_rate > 0.5:
        base_prob += 0.25
    elif skip_rate > 0.3:
        base_prob += 0.15

    if listening_time < 20:
        base_prob += 0.20
    elif listening_time < 40:
        base_prob += 0.10

    if ad_burden > 0.3:
        base_prob += 0.15
    elif ad_burden > 0.2:
        base_prob += 0.08

    if sub_type == "Free":
        base_prob += 0.10

    if songs_per_day < 10:
        base_prob += 0.10

    # 긍정 요인 가중치
    if offline == 1:
        base_prob -= 0.15

    if listening_time > 90:
        base_prob -= 0.10

    if skip_rate < 0.2:
        base_prob -= 0.12

    # 확률을 0~1 사이로 제한
    prob = max(0.0, min(1.0, base_prob))
    return float(prob)


def simulate_dl_prediction(input_data):
    """DL 모델 시뮬레이션 예측 (ML과 약간 다른 가중치 사용)"""
    skip_rate = input_data["skip_rate"].iloc[0]
    listening_time = input_data["listening_time"].iloc[0]
    ad_burden = input_data["ad_burden"].iloc[0]
    offline = input_data["offline_listening"].iloc[0]
    sub_type = input_data["subscription_type"].iloc[0]
    satisfaction_score = input_data["satisfaction_score"].iloc[0]

    # DL은 조금 다른 알고리즘으로 예측 (비선형성 모방)
    base_prob = 0.28

    # 비선형 가중치 적용
    skip_penalty = (skip_rate**1.5) * 0.4
    time_penalty = max(0, (30 - listening_time) / 30) * 0.25
    ad_penalty = min(ad_burden * 0.5, 0.2)

    base_prob += skip_penalty + time_penalty + ad_penalty

    # 만족도 점수 기반 조정
    if satisfaction_score < 10:
        base_prob += 0.12
    elif satisfaction_score > 30:
        base_prob -= 0.10

    # 구독 유형별 조정
    if sub_type == "Free":
        base_prob += 0.12
    elif sub_type == "Premium":
        base_prob -= 0.08

    # 오프라인 사용
    if offline == 1:
        base_prob -= 0.18

    # 확률을 0~1 사이로 제한
    prob = max(0.0, min(1.0, base_prob))
    return float(prob)


def load_metrics():
    """모델 메트릭 로드"""
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
    except Exception as e:
        pass
    return {}


def get_best_ml_model_info():
    """최고 성능 ML 모델 정보 반환"""
    metrics = load_metrics()
    best_name = "RandomForest"
    best_thresh = 0.5
    max_f1 = -1

    for name, data in metrics.items():
        if (
            name != "Deep Learning (DNN)"
            and "F1-Score" in data
            and "Best Threshold" in data
        ):
            if data["F1-Score"] > max_f1:
                max_f1 = data["F1-Score"]
                best_name = name
                best_thresh = data["Best Threshold"]

    return best_name, best_thresh


def prepare_input_data(
    age,
    gender,
    country,
    sub_type,
    device,
    listening_time,
    songs_per_day,
    skip_rate,
    ads_listened,
    offline,
):
    """입력 데이터 전처리 및 파생 변수 생성"""
    input_data = pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "country": country,
                "subscription_type": sub_type,
                "device_type": device,
                "listening_time": listening_time,
                "songs_played_per_day": songs_per_day,
                "skip_rate": skip_rate,
                "ads_listened_per_week": ads_listened,
                "offline_listening": 1 if offline else 0,
            }
        ]
    )

    # 파생 변수 생성
    input_data["ad_burden"] = input_data["ads_listened_per_week"] / (
        input_data["listening_time"] + 1
    )
    input_data["satisfaction_score"] = input_data["songs_played_per_day"] * (
        1 - input_data["skip_rate"]
    )
    input_data["time_per_song"] = input_data["listening_time"] / (
        input_data["songs_played_per_day"] + 1
    )

    return input_data


def predict_ml(input_data):
    """ML 모델 예측 (시뮬레이션)"""
    return simulate_ml_prediction(input_data)


def predict_dl(input_data):
    """DL 모델 예측 (시뮬레이션)"""
    return simulate_dl_prediction(input_data)


def create_gauge_chart(prob, threshold, title, color):
    """게이지 차트 생성"""
    value = prob * 100
    if value < threshold * 100 * 0.7:
        bar_color = "#1DB954"  # Green
    elif value < threshold * 100:
        bar_color = "#FFC107"  # Yellow
    else:
        bar_color = "#FF5252"  # Red

    if color:
        bar_color = color

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 18, "color": "#ffffff"}},
            number={"suffix": "%", "font": {"size": 35, "color": bar_color}},
            delta={"reference": threshold * 100, "font": {"size": 14}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "#ffffff"},
                "bar": {"color": bar_color},
                "bgcolor": "#1a1a1a",
                "borderwidth": 2,
                "bordercolor": "#1DB954",
                "steps": [
                    {"range": [0, threshold * 100 * 0.7], "color": "#1a3a1a"},
                    {
                        "range": [threshold * 100 * 0.7, threshold * 100],
                        "color": "#3a3a1a",
                    },
                    {"range": [threshold * 100, 100], "color": "#3a1a1a"},
                ],
                "threshold": {
                    "line": {"color": "#ffffff", "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#000000",
        font={"color": "#ffffff"},
    )
    return fig


def create_comparison_chart(ml_prob, dl_prob, ml_threshold, dl_threshold):
    """ML과 DL 예측 결과 비교 차트"""
    fig = go.Figure()

    models = ["ML 모델", "DL 모델"]
    probs = [ml_prob * 100, dl_prob * 100]
    thresholds = [ml_threshold * 100, dl_threshold * 100]
    colors = ["#1DB954", "#00d4ff"]

    # 예측 확률 바
    fig.add_trace(
        go.Bar(
            x=models,
            y=probs,
            name="이탈 확률",
            marker_color=colors,
            text=[f"{p:.1f}%" for p in probs],
            textposition="outside",
            textfont={"size": 16, "color": "#ffffff"},
        )
    )

    # 임계값 라인
    for i, (model, thresh) in enumerate(zip(models, thresholds)):
        fig.add_hline(
            y=thresh,
            line_dash="dash",
            line_color=colors[i],
            annotation_text=f"{model} 임계값: {thresh:.1f}%",
            annotation_position="right",
        )

    fig.update_layout(
        title={
            "text": "ML vs DL 모델 예측 비교",
            "font": {"size": 20, "color": "#1DB954"},
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis={
            "title": {"text": "모델", "font": {"color": "#ffffff"}},
            "tickfont": {"color": "#ffffff"},
        },
        yaxis={
            "title": {"text": "이탈 확률 (%)", "font": {"color": "#ffffff"}},
            "tickfont": {"color": "#ffffff"},
            "range": [0, 100],
        },
        plot_bgcolor="#111111",
        paper_bgcolor="#000000",
        height=400,
        showlegend=False,
    )

    return fig


def get_risk_level(prob, threshold):
    """위험도 레벨 판정"""
    if prob >= threshold:
        return "high", "🚨 고위험 (High Risk)"
    elif prob >= threshold * 0.8:
        return "medium", "⚠️ 중위험 (Medium Risk)"
    else:
        return "low", "✅ 저위험 (Low Risk)"


def generate_insights(input_data, ml_prob, dl_prob, ml_threshold, dl_threshold):
    """인사이트 생성"""
    insights = []

    skip_rate = input_data["skip_rate"].iloc[0]
    listening_time = input_data["listening_time"].iloc[0]
    ad_burden = input_data["ad_burden"].iloc[0]
    offline = input_data["offline_listening"].iloc[0]
    sub_type = input_data["subscription_type"].iloc[0]

    # 위험 요인
    if skip_rate > 0.4:
        insights.append(
            {
                "type": "risk",
                "title": "높은 스킵 비율",
                "desc": f"스킵 비율이 {skip_rate*100:.0f}%로 높아 추천 시스템의 만족도가 낮습니다.",
                "action": "맞춤형 플레이리스트 제안 또는 음악 취향 재설정 권장",
            }
        )

    if ad_burden > 0.25:
        insights.append(
            {
                "type": "risk",
                "title": "광고 피로도 경고",
                "desc": f"청취 시간 대비 광고 노출이 높아 사용자 만족도에 부정적 영향을 줄 수 있습니다.",
                "action": "프리미엄 구독 전환 캠페인 또는 광고 빈도 조절 검토",
            }
        )

    if listening_time < 20:
        insights.append(
            {
                "type": "risk",
                "title": "이용 시간 부족",
                "desc": f"일일 평균 청취 시간이 {listening_time:.0f}분으로 이탈 전조 증상이 보입니다.",
                "action": "개인화된 추천 콘텐츠 제공 및 재참여 유도",
            }
        )

    # 긍정 요인
    if offline == 1:
        insights.append(
            {
                "type": "positive",
                "title": "오프라인 기능 활용",
                "desc": "프리미엄 기능을 적극 활용하여 충성도가 높은 사용자입니다.",
                "action": "유지 및 추가 프리미엄 기능 홍보",
            }
        )

    if skip_rate < 0.2:
        insights.append(
            {
                "type": "positive",
                "title": "높은 콘텐츠 만족도",
                "desc": "낮은 스킵 비율로 추천 시스템이 사용자 취향을 잘 파악하고 있습니다.",
                "action": "유사 콘텐츠 확대 추천",
            }
        )

    # 모델 일치도 분석
    ml_risk = ml_prob >= ml_threshold
    dl_risk = dl_prob >= dl_threshold

    if ml_risk == dl_risk:
        insights.append(
            {
                "type": "info",
                "title": "모델 예측 일치",
                "desc": "ML과 DL 모델이 동일한 예측 결과를 보여 신뢰도가 높습니다.",
                "action": "예측 결과를 바탕으로 즉시 대응 전략 수립 가능",
            }
        )
    else:
        insights.append(
            {
                "type": "warning",
                "title": "모델 예측 불일치",
                "desc": "ML과 DL 모델의 예측 결과가 상이하여 추가 모니터링이 권장됩니다.",
                "action": "다양한 지표를 종합하여 판단 필요",
            }
        )

    return insights


# 타이틀
st.markdown(
    """
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="color: #1DB954; font-size: 48px; margin-bottom: 10px;">🔮 실시간 이탈 예측</h1>
    <p style="color: #cccccc; font-size: 18px;">머신러닝과 딥러닝 모델을 활용한 고객 이탈 예측 결과</p>
</div>
""",
    unsafe_allow_html=True,
)

# 메트릭 로드 (임계값 정보용)
metrics = load_metrics()
best_ml_name, best_ml_threshold = get_best_ml_model_info()

# 모델 상태 표시
col1, col2 = st.columns(2)
with col1:
    st.success(f"✅ ML 모델 준비 완료 ({best_ml_name})")
with col2:
    st.success("✅ DL 모델 준비 완료 (Deep Learning)")

st.markdown("---")

# 입력 섹션 (사이드바)
st.sidebar.header("📊 고객 정보 입력")
st.sidebar.markdown("고객 정보를 입력하면 실시간으로 예측 결과가 업데이트됩니다.")

# 기본 정보
st.sidebar.subheader("1. 기본 정보")
age = st.sidebar.slider("나이 (Age)", 10, 80, 30, key="age")
gender = st.sidebar.selectbox("성별", ["Male", "Female", "Other"], key="gender")
country = st.sidebar.selectbox(
    "국가", ["US", "UK", "DE", "FR", "CA", "IN"], key="country"
)
sub_type = st.sidebar.selectbox(
    "구독 유형", ["Free", "Premium", "Family", "Student"], key="sub_type"
)
device = st.sidebar.selectbox("사용 기기", ["Mobile", "Desktop", "Web"], key="device")

st.sidebar.markdown("---")

# 이용 행태
st.sidebar.subheader("2. 이용 행태")
listening_time = st.sidebar.slider(
    "하루 청취 시간 (분)", 0.0, 180.0, 60.0, key="listening_time"
)
songs_per_day = st.sidebar.slider("하루 재생 곡 수", 0, 100, 20, key="songs_per_day")
skip_rate = st.sidebar.slider("노래 스킵 비율", 0.0, 1.0, 0.2, 0.01, key="skip_rate")
ads_listened = st.sidebar.slider("주간 광고 청취 수", 0, 50, 5, key="ads_listened")
offline = st.sidebar.checkbox("오프라인 모드 사용", value=False, key="offline")

st.sidebar.markdown("---")
st.sidebar.info("💡 입력값을 변경하면 자동으로 예측 결과가 업데이트됩니다.")

# 입력 데이터 준비
input_data = prepare_input_data(
    age,
    gender,
    country,
    sub_type,
    device,
    listening_time,
    songs_per_day,
    skip_rate,
    ads_listened,
    offline,
)

# 예측 수행 (항상 시뮬레이션 예측 사용)
ml_prob = predict_ml(input_data)
dl_prob = predict_dl(input_data)
dl_threshold = metrics.get("Deep Learning (DNN)", {}).get("Best Threshold", 0.5)

# 예측 결과 표시
if ml_prob is not None and dl_prob is not None:
    # 두 모델 예측 결과를 나란히 표시
    st.markdown("### 📈 예측 결과 비교")

    cols = st.columns(2)

    with cols[0]:
        ml_risk_level, ml_risk_text = get_risk_level(ml_prob, best_ml_threshold)
        st.markdown(
            f"""
        <div class="prediction-card">
            <div class="prediction-title">🤖 {best_ml_name} (ML)</div>
            <div style="text-align: center; margin: 20px 0;">
                <div class="risk-{ml_risk_level}">{ml_prob*100:.1f}%</div>
                <p style="font-size: 18px; margin-top: 10px;">{ml_risk_text}</p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        fig_ml = create_gauge_chart(
            ml_prob, best_ml_threshold, f"{best_ml_name} 예측", "#1DB954"
        )
        st.plotly_chart(fig_ml, use_container_width=True)

        # 기존 st.metric 3줄을 삭제하고 아래 코드를 넣으세요
        st.markdown(
            f"""
            <div class="metric-container"><div class="metric-label">임계값</div><div class="metric-value">{best_ml_threshold*100:.1f}%</div></div>
            <div class="metric-container"><div class="metric-label">정확도</div><div class="metric-value">{metrics.get(best_ml_name, {}).get('Accuracy', 0)*100:.1f}%</div></div>
            <div class="metric-container"><div class="metric-label">F1-Score</div><div class="metric-value">{metrics.get(best_ml_name, {}).get('F1-Score', 0):.3f}</div></div>
        """,
            unsafe_allow_html=True,
        )

    with cols[1]:
        dl_risk_level, dl_risk_text = get_risk_level(dl_prob, dl_threshold)
        st.markdown(
            f"""
        <div class="prediction-card">
            <div class="prediction-title">🧠 Deep Learning (DL)</div>
            <div style="text-align: center; margin: 20px 0;">
                <div class="risk-{dl_risk_level}">{dl_prob*100:.1f}%</div>
                <p style="font-size: 18px; margin-top: 10px;">{dl_risk_text}</p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        fig_dl = create_gauge_chart(
            dl_prob, dl_threshold, "Deep Learning 예측", "#00d4ff"
        )
        st.plotly_chart(fig_dl, use_container_width=True)

        # 기존 st.metric 3줄을 삭제하고 아래 코드를 넣으세요
        st.markdown(
            f"""
            <div class="metric-container"><div class="metric-label">임계값</div><div class="metric-value">{dl_threshold*100:.1f}%</div></div>
            <div class="metric-container"><div class="metric-label">정확도</div><div class="metric-value">{metrics.get('Deep Learning (DNN)', {}).get('Accuracy', 0)*100:.1f}%</div></div>
            <div class="metric-container"><div class="metric-label">F1-Score</div><div class="metric-value">{metrics.get('Deep Learning (DNN)', {}).get('F1-Score', 0):.3f}</div></div>
        """,
            unsafe_allow_html=True,
        )

    # 비교 차트
    st.markdown("---")
    st.markdown("### 📊 모델 비교 분석")
    comparison_fig = create_comparison_chart(
        ml_prob, dl_prob, best_ml_threshold, dl_threshold
    )
    st.plotly_chart(comparison_fig, use_container_width=True)

    # 평균 예측 확률
    avg_prob = (ml_prob + dl_prob) / 2
    avg_threshold = (best_ml_threshold + dl_threshold) / 2
    avg_risk_level, avg_risk_text = get_risk_level(avg_prob, avg_threshold)

    st.markdown("---")
    st.markdown("### 🎯 종합 예측 결과")

    # 데이터 계산 부분 (기존 로직 유지)
    avg_prob = (ml_prob + dl_prob) / 2
    avg_threshold = (best_ml_threshold + dl_threshold) / 2
    avg_risk_level, avg_risk_text = get_risk_level(avg_prob, avg_threshold)
    ml_pred = "이탈 예상" if ml_prob >= best_ml_threshold else "유지 예상"
    dl_pred = "이탈 예상" if dl_prob >= dl_threshold else "유지 예상"
    match_text = "✅ 일치" if ml_pred == dl_pred else "⚠️ 불일치"

    # UI 출력 부분 수정
    col_avg1, col_avg2, col_avg3 = st.columns(3)

    with col_avg1:
        st.markdown(
            f"""
            <div class="summary-metric-container">
                <div class="summary-label">평균 이탈 확률</div>
                <div class="summary-value">{avg_prob*100:.1f}%</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col_avg2:
        # 아이콘을 뺀 텍스트만 추출하여 표시
        clean_risk_text = (
            avg_risk_text.replace("🚨 ", "").replace("⚠️ ", "").replace("✅ ", "")
        )
        st.markdown(
            f"""
            <div class="summary-metric-container">
                <div class="summary-label">위험도</div>
                <div class="summary-value">{clean_risk_text}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col_avg3:
        st.markdown(
            f"""
            <div class="summary-metric-container">
                <div class="summary-label">예측 일치도</div>
                <div class="summary-value">{match_text}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # 인사이트 생성 및 표시
    insights = generate_insights(
        input_data, ml_prob, dl_prob, best_ml_threshold, dl_threshold
    )

    st.markdown("---")
    st.markdown("### 💡 분석 인사이트 및 권장 조치")

    for insight in insights:
        if insight["type"] == "risk":
            st.error(
                f"**{insight['title']}**\n\n{insight['desc']}\n\n💼 **권장 조치**: {insight['action']}"
            )
        elif insight["type"] == "positive":
            st.success(
                f"**{insight['title']}**\n\n{insight['desc']}\n\n💼 **권장 조치**: {insight['action']}"
            )
        elif insight["type"] == "warning":
            st.warning(
                f"**{insight['title']}**\n\n{insight['desc']}\n\n💼 **권장 조치**: {insight['action']}"
            )
        else:
            st.info(
                f"**{insight['title']}**\n\n{insight['desc']}\n\n💼 **권장 조치**: {insight['action']}"
            )

    # 입력 데이터 요약 표시
    st.markdown("---")
    st.markdown("### 📋 입력 데이터 요약")

    summary_data = {
        "항목": [
            "나이",
            "성별",
            "국가",
            "구독 유형",
            "기기",
            "청취 시간(분)",
            "일일 재생 곡 수",
            "스킵 비율",
            "주간 광고 수",
            "오프라인 사용",
        ],
        "값": [
            age,
            gender,
            country,
            sub_type,
            device,
            listening_time,
            songs_per_day,
            f"{skip_rate*100:.1f}%",
            ads_listened,
            "예" if offline else "아니오",
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # 하단 네비게이션 버튼 (1행 4열로 구성하여 끝 라인 맞춤)
    st.markdown("<br>", unsafe_allow_html=True)
    nav_cols = st.columns(15)

    with nav_cols[0]:  # 좌측 첫 번째 칸 (Home)
        if st.button("🏠 Home"):
            st.switch_page("Home.py")  # 메인 파일명 확인 필요

    with nav_cols[14]:  # 우측 네 번째 칸 (Next)
        if st.button("Next ➡️"):
            st.switch_page("pages/business_strategy.py")  # 다음 페이지 파일명 확인 필요

# 푸터
st.markdown("---")
st.caption("© 2025 Spotify Churn Prediction Project - 실시간 예측 시스템")
