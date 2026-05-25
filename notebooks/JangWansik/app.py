import streamlit as st
import pandas as pd
import joblib
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os
import tensorflow as tf

# --- [설정] 파일 경로 정의 ---
# 모델과 메트릭 파일이 위치한 폴더명입니다.
MODEL_DIR = '03_trained_model'

st.set_page_config(
    page_title="Spotify Churn Insight AI",
    page_icon="🎵",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main { background-color: #F0F2F6; }
    .stButton>button {
        background-color: #1DB954; color: white; border-radius: 20px; border: none; font-weight: bold; width: 100%; height: 50px; font-size: 18px;
    }
    div[role="radiogroup"] > label > div:first-child {
        background-color: #1DB954 !important; color: #1DB954 !important; border-color: #1DB954 !important;
    }
    .report-card {
        background-color: white; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); padding: 0px; margin-top: 20px; overflow: hidden;
    }
    .report-header {
        background-color: #1DB954; padding: 15px 25px; color: white; font-size: 20px; font-weight: bold; border-bottom: 1px solid #e0e0e0;
    }
    .report-body { padding: 25px; }
    .legend-box {
        background-color: #f8f9fa; border-radius: 8px; padding: 10px; margin-bottom: 20px; text-align: center; font-size: 14px; color: #333 !important; border: 1px solid #eee;
    }
    .factor-bar {
        padding: 15px; margin-bottom: 12px; border-radius: 8px; color: black !important; font-weight: 500; display: flex; align-items: center;
    }
    .risk { background-color: #ffebee; border-left: 6px solid #ff5252; }
    .complex { background-color: #fff3e0; border-left: 6px solid #ff9800; }
    .safe { background-color: #e8f5e9; border-left: 6px solid #4caf50; }
    .info-box {
        background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 15px; margin-top: 10px; font-size: 14px; color: #0d47a1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_ml_model():
    try:
        # [수정됨] 경로를 MODEL_DIR 상수로 변경
        model_path = os.path.join(MODEL_DIR, 'spotify_churn_model.pkl')
        return joblib.load(model_path)
    except:
        return None

@st.cache_resource
def load_dl_model_and_scaler():
    model = None
    scaler = None
    
    try:
        # [수정됨] 경로를 MODEL_DIR 상수로 변경
        model_path = os.path.join(MODEL_DIR, 'spotify_dl_model.h5')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
    except Exception as e:
        #st.error(f"DL 모델 로딩 실패: {e}")
        pass

    try:
        # [수정됨] 경로를 MODEL_DIR 상수로 변경
        scaler_path = os.path.join(MODEL_DIR, 'dl_preprocessor.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
    except Exception as e:
        #st.error(f"전처리기 로딩 실패: {e}")
        pass
        
    return model, scaler

def load_metrics():
    try:
        # [수정됨] model_metrics.json 파일도 03_trained_model 폴더로 이동했으므로 경로 수정
        metrics_path = os.path.join(MODEL_DIR, 'model_metrics.json')
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except:
        return {}

# ML 모델 중 F1-Score가 가장 높은 모델을 찾습니다. (Prediction 페이지의 ML 옵션용)
def get_best_model_info():
    metrics = load_metrics()
    best_name = "Optimized ML Model"
    best_thresh = 0.5
    max_f1 = -1
    
    # ML 모델(DNN 제외) 중 F1-Score가 가장 높고 Best Threshold가 있는 모델을 찾음
    for name, data in metrics.items():
        if name != "Deep Learning (DNN)" and 'F1-Score' in data and 'Best Threshold' in data:
            if data['F1-Score'] > max_f1:
                max_f1 = data['F1-Score']
                best_name = name
                best_thresh = data['Best Threshold']
    
    # 만약 유효한 ML 모델이 없으면 RandomForest의 임계값으로 대체 (안전장치)
    if max_f1 == -1 and 'RandomForest' in metrics and 'Best Threshold' in metrics['RandomForest']:
         best_thresh = metrics['RandomForest']['Best Threshold']
         
    return best_name, best_thresh

# [절대적인 최고 성능 모델을 찾는 함수] - Dashboard의 추천 모델 선정에 사용
def get_absolute_best_model_name():
    metrics = load_metrics()
    best_name = "최고 성능 모델"
    max_f1 = -1
    
    for name, data in metrics.items():
        if 'F1-Score' in data:
            if data['F1-Score'] > max_f1:
                max_f1 = data['F1-Score']
                best_name = name
                
    return best_name


def make_radar_chart(input_data):
    # 특성 값 정규화/스케일링 로직
    # 참고: ad_burden, skip_rate 등은 0-1 사이로 스케일링
    immersion = min(input_data['listening_time'][0] / 60, 1.0) * 100
    satisfaction = (1 - input_data['skip_rate'][0]) * 100
    activity = min(input_data['songs_played_per_day'][0] / 30, 1.0) * 100
    ad_burden = input_data['ad_burden'][0]
    tolerance = max(0, (1 - min(ad_burden * 3, 1.0))) * 100 # ad_burden이 높을수록 tolerance 낮아짐
    
    loyalty = 50
    if input_data['offline_listening'][0] == 1: loyalty += 30
    if input_data['subscription_type'][0] != 'Free': loyalty += 20
    
    categories = ['몰입도(시간)', '만족도(No Skip)', '활동성(곡 수)', '광고 내성', '충성도']
    values = [immersion, satisfaction, activity, tolerance, loyalty]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself', name='User Profile', line_color='#1DB954'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, margin=dict(l=40, r=40, t=30, b=30), height=300
    )
    return fig

def make_gauge_chart(prob, threshold):
    value = prob * 100
    if value < 40: bar_color = "#1DB954" 
    elif value < threshold * 100: bar_color = "#FFC107" 
    else: bar_color = "#FF5252" 

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "이탈 확률 (%)", 'font': {'size': 20}},
        number = {'suffix': "%", 'font': {'size': 40, 'color': bar_color}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': "#e8f5e9"},
                {'range': [40, threshold*100], 'color': "#fff3e0"},
                {'range': [threshold*100, 100], 'color': "#ffebee"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def page_home():
    st.title("🎵 Spotify 이탈 예측 프로젝트 개요")
    
    st.markdown("""
    ### <span style='color:#1DB954'>프로젝트 소개</span>
    이 프로젝트는 스포티파이 사용자 데이터를 분석하여 **사용자 이탈 가능성을 예측**하고,
    이를 바탕으로 **비즈니스 인사이트와 대응 전략**을 제시하는 AI 서비스 데모입니다.

    머신러닝(ML)과 딥러닝(DL) 모델을 활용하여 고객의 행동 패턴을 분석하고, 
    이탈 위험이 높은 사용자를 조기에 식별하여 맞춤형 관리를 할 수 있도록 돕습니다.

    ---

    ### <span style='color:#1DB954'>주요 기능</span>
    
    #### 1. 📊 모델 성능 비교
    * 학습된 다양한 AI 모델(RandomForest, XGBoost, Deep Learning 등)의 성능 지표(정확도, F1-Score)를 시각적으로 비교 분석합니다.
    * 가장 우수한 성능을 보인 'Best Model' 선정 근거를 확인할 수 있습니다.

    #### 2. 🔮 실전 이탈 예측 & 심층 분석
    * 사용자의 나이, 구독 정보, 청취 습관 등의 데이터를 입력하면 AI가 실시간으로 이탈 확률을 진단합니다.
    * **레이더 차트**를 통해 유저 성향을 파악하고, **상세 분석 리포트**를 통해 이탈 위험 요인과 긍정 요인을 파악할 수 있습니다.
    * ML 모델과 DL 모델 중 원하는 모델을 선택하여 예측 결과를 비교해 볼 수 있습니다.

    #### 3. 💡 비즈니스 인사이트
    * 예측된 이탈 확률 구간별(위험/경고/안정)로 맞춤형 비즈니스 액션 플랜을 제안합니다.
    * 데이터 분석을 통해 도출된 전반적인 서비스 개선 방향(Product Insight)을 제공합니다.
    
    ---
    <br>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Team Info**\n\nSKN22-2nd-3Team Project")
    with col2:
        st.success("**Data Source**\n\n[Kaggle Spotify Churn Dataset](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis/)")

def page_dashboard():
    st.title("📊 모델 성능 비교 대시보드")
    col1, col2 = st.columns([2, 1])
    
    # [최고 성능 모델 동적 선정]
    absolute_best_model_name = get_absolute_best_model_name()
    
    with col1:
        st.subheader("모델별 정확도(Accuracy) & F1-Score")
        metrics = load_metrics()
        
        if metrics:
            model_names = list(metrics.keys())
            acc_scores = [metrics[m]['Accuracy'] for m in model_names]
            f1_scores = [metrics[m]['F1-Score'] for m in model_names]
            
            df_plot = pd.DataFrame({
                'Model': model_names * 2,
                'Score': acc_scores + f1_scores,
                'Metric': ['Accuracy'] * len(model_names) + ['F1-Score'] * len(model_names)
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", palette="viridis", ax=ax)
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)
            plt.ylim(0.5, 1.05)
            plt.xticks(rotation=15)
            st.pyplot(fig)
        else:
            st.error("모델 성능 파일(model_metrics.json)을 찾을 수 없습니다.")
            
    with col2:
        st.info("💡 모델 선정 분석")
        if metrics:
            st.markdown(f"""
            **🏆 추천 모델: {absolute_best_model_name}**
            
            **선정 이유:**
            1. **최고 성능:** 후보 모델 중 F1-Score가 가장 높아 이탈 사용자 탐지에 가장 효과적임
            2. **안정성:** 과적합 위험이 적음
            3. **효율성:** 실시간 예측에 적합
            """)

def page_prediction():
    st.title("🔮 실전 이탈 예측 & 심층 분석")
    
    # ML/DL 각각의 정보를 로드
    best_ml_name, best_ml_threshold = get_best_model_info()
    absolute_best_name = get_absolute_best_model_name() # 현재 최고 성능 모델 이름
    
    st.sidebar.header("1. 사용자 정보 입력")
    age = st.sidebar.slider("나이 (Age)", 10, 80, 25)
    gender = st.sidebar.selectbox("성별", ["Male", "Female", "Other"])
    sub_type = st.sidebar.selectbox("구독 유형", ["Free", "Premium", "Family", "Student"])
    device = st.sidebar.selectbox("사용 기기", ["Mobile", "Desktop", "Web"])
    
    st.sidebar.markdown("---")
    st.sidebar.header("2. 이용 행태 정보")
    listening_time = st.sidebar.slider("하루 청취 시간 (분)", 0.0, 180.0, 60.0)
    songs_per_day = st.sidebar.slider("하루 재생 곡 수", 0, 100, 20)
    skip_rate = st.sidebar.slider("노래 스킵 비율 (Skip Rate)", 0.0, 1.0, 0.2)
    ads_listened = st.sidebar.slider("주간 광고 청취 수", 0, 50, 5)
    offline = st.sidebar.checkbox("오프라인 모드 사용", value=False)
    
    st.sidebar.markdown("---")
    
    st.sidebar.header("3. 모델 선택")
    
    ml_label = f"{best_ml_name} (ML)" 
    dl_label = "Deep Learning (DNN)"
    
    # [수정된 로직]: 최고 성능 모델이 DNN일 경우, DNN을 기본 선택(index=0)으로 설정
    if absolute_best_name == "Deep Learning (DNN)":
        model_options = [dl_label, ml_label]
        default_index = 0
    else:
        model_options = [ml_label, dl_label]
        default_index = 0
        
    model_choice = st.sidebar.radio("예측에 사용할 모델을 선택하세요.", model_options, index=default_index)
    
    st.sidebar.write("")
    predict_btn = st.sidebar.button("분석 시작")

    if predict_btn:
        # 1. 입력 데이터 전처리 (파생 변수 생성)
        input_data = pd.DataFrame([{
            'age': age,
            'gender': gender,
            'listening_time': listening_time,
            'songs_played_per_day': songs_per_day,
            'skip_rate': skip_rate,
            'ads_listened_per_week': ads_listened,
            'country': 'US',
            'subscription_type': sub_type,
            'device_type': device,
            'offline_listening': 1 if offline else 0
        }])
        
        # 파생 변수 (EDA 및 전처리 단계에서 도출된 변수)
        input_data['ad_burden'] = input_data['ads_listened_per_week'] / (input_data['listening_time'] + 1)
        input_data['satisfaction_score'] = input_data['songs_played_per_day'] * (1 - input_data['skip_rate'])
        input_data['time_per_song'] = input_data['listening_time'] / (input_data['songs_played_per_day'] + 1)
        
        prob = 0.5
        threshold = 0.5
        
        # 2. 선택된 모델로 예측 수행
        if model_choice == ml_label:
            model = load_ml_model()
            if model:
                # ML 모델 (RandomForest/XGBoost 등) 예측
                prob = model.predict_proba(input_data)[0, 1]
                threshold = best_ml_threshold
            else:
                st.error("ML 모델 파일(.pkl)을 찾을 수 없습니다.")

        elif model_choice == dl_label:
            dl_model, dl_scaler = load_dl_model_and_scaler()
            
            if dl_model and dl_scaler:
                try:
                    # DL 모델 전처리 및 예측
                    # dl_scaler는 학습 시 사용된 모든 컬럼을 처리한다고 가정
                    scaled_input = dl_scaler.transform(input_data)
                    prediction = dl_model.predict(scaled_input)
                    prob = float(prediction[0][0])
                    
                    metrics = load_metrics()
                    # DL 모델의 최적 임계값 로드
                    threshold = metrics.get('Deep Learning (DNN)', {}).get('Best Threshold', 0.5)
                    
                except Exception as e:
                    st.error(f"DL 예측 중 오류 발생: {e}")
            else:
                st.error("DL 모델(.h5) 또는 전처리기(.pkl)를 불러올 수 없습니다.")

        st.markdown("### 🎯 AI 예측 진단")
        
        col1, col2, col3 = st.columns([1, 2, 2])
        
        # 3. 예측 결과 시각화 및 리포트
        with col1:
            st.write("") 
            st.write("") 
            if prob >= threshold:
                st.error("🚨 **위험 (High)**")
                st.write(f"이탈 확률 {threshold:.2f} 기준 초과")
            else:
                st.success("✅ **안전 (Safe)**")
                st.write(f"이탈 확률 {threshold:.2f} 기준 미만")
                
        with col2:
            fig_gauge = make_gauge_chart(prob, threshold)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col3:
            st.write("🕸️ **유저 프로필 분석**")
            fig = make_radar_chart(input_data)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # 4. 상세 분석 리포트 생성 (규칙 기반)
        negative_factors = [] 
        complex_factors = []
        positive_factors = []
        
        if skip_rate > 0.4: 
            negative_factors.append(f"<b>높은 스킵 비율({skip_rate*100:.0f}%)</b>: 추천 곡 불만족")
        if input_data['ad_burden'][0] > 0.25: 
            negative_factors.append("<b>광고 피로도 경고</b>: 청취 시간 대비 잦은 광고 (Free User)")
        if listening_time < 20: 
            negative_factors.append(f"<b>이용 시간 부족({listening_time}분)</b>: 이탈 전조 증상")
        
        if listening_time > 60 and skip_rate > 0.5:
            complex_factors.append("<b>📉 '풍요 속의 빈곤' 패턴</b>: 사용량은 많지만 만족도가 낮음")
        if listening_time < 30 and input_data['ad_burden'][0] > 0.3:
            complex_factors.append("<b>⚡ '광고 충격' 패턴</b>: 짧게 듣고 광고만 듣다 나감")

        if offline: 
            positive_factors.append("<b>오프라인 기능 활용</b>: 충성도 높음 (Premium)")
        if skip_rate < 0.2: 
            positive_factors.append("<b>취향 저격 성공</b>: 낮은 스킵률")

        if prob >= threshold and not negative_factors and not complex_factors:
            complex_factors.append("<b>🧩 잠재적 복합 위험군</b>: 여러 행동 패턴이 복합적으로 '이탈'을 가리킴")

        with st.container():
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.markdown('<div class="report-header">📝 AI 상세 분석 리포트</div>', unsafe_allow_html=True)
            st.markdown('<div class="report-body">', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="legend-box">
                <span style="color:#ff5252"><b>🟥 위험 요인</b></span> &nbsp;|&nbsp; 
                <span style="color:#ff9800"><b>🟧 복합/심층 원인</b></span> &nbsp;|&nbsp; 
                <span style="color:#4caf50"><b>🟩 긍정 요인</b></span>
            </div>
            """, unsafe_allow_html=True)

            if any("잠재적 복합 위험군" in s for s in complex_factors):
                st.markdown("""
                <div class="info-box">
                    <b>❓ '잠재적 복합 위험군'이란?</b><br>
                    특정한 하나의 문제(예: 스킵 과다)가 뚜렷하지 않지만, 나이, 구독 형태, 청취 패턴 등 
                    <b>여러 요소가 미세하게 얽혀 AI가 이탈 가능성을 높게 판단한 그룹</b>입니다. 
                    이들은 불만을 표출하지 않고 조용히 서비스를 떠나는 <b>'Silent Churner'</b>일 확률이 높습니다.
                </div>
                """, unsafe_allow_html=True)

            if negative_factors:
                for f in negative_factors: st.markdown(f'<div class="factor-bar risk">🚨 {f}</div>', unsafe_allow_html=True)
            if complex_factors:
                for f in complex_factors: st.markdown(f'<div class="factor-bar complex">🕵️ {f}</div>', unsafe_allow_html=True)
            if positive_factors:
                for f in positive_factors: st.markdown(f'<div class="factor-bar safe">💚 {f}</div>', unsafe_allow_html=True)
            
            if not negative_factors and not complex_factors and not positive_factors:
                 st.markdown('<div class="factor-bar safe">✅ <b>특이 사항 없음:</b> 안정적인 패턴입니다.</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

def page_insights():
    st.title("💡 비즈니스 전략 가이드")
    st.markdown("### 📌 AI 분석 기반 액션 플랜")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔴 긴급 방어 (Risk)", "🟡 잠재 관리 (Warning)", "🟢 충성/수익화 (Loyal)", "⚙️ 서비스 개선 (Product)"])
    
    with tab1:
        st.markdown("#### 🚨 이탈 확률 70% 이상: 즉각적인 개입 필요")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**💰 가격 방어 전략**")
            st.write("- **시크릿 오퍼:** 3개월 50% 할인 쿠폰 즉시 푸시 발송")
            st.write("- **다운그레이드 제안:** 해지 대신 '광고형 무료 요금제' 유지 유도")
        with col2:
            st.info("**🎧 콘텐츠 심폐소생**")
            st.write("- **향수 마케팅:** 'OO님이 2년 전 가장 많이 들었던 곡' 플레이리스트 생성")
            st.write("- **큐레이션 리셋:** 기존 추천 알고리즘 초기화 옵션 제공")

    with tab2:
        st.markdown("#### ⚠️ 이탈 확률 40~70%: 골든타임 관리")
        st.write("이 그룹은 아직 서비스를 이용 중이지만, 불만이 쌓이고 있습니다. 'Silent Churn'을 막아야 합니다.")
        st.markdown("""
        * **광고 피로도 관리:** Free 유저의 경우, 향후 2주간 **광고 노출 빈도를 30% 축소**하여 사용자 경험 개선
        * **기능 튜토리얼:** '데이터 절약 모드', '오프라인 저장' 등 유용한 기능을 팝업으로 안내하여 앱 효용성 증대
        * **푸시 알림 최적화:** 맹목적인 알림 대신, 선호 아티스트의 신곡 알림만 선별 발송
        """)

    with tab3:
        st.markdown("#### 💎 이탈 확률 40% 미만: 수익 극대화 및 락인(Lock-in)")
        col1, col2 = st.columns(2)
        with col1:
            st.success("**💸 Upselling (객단가 상승)**")
            st.write("- **패밀리/듀오 요금제:** 혼자 쓰는 유저에게 '친구와 함께 쓰면 반값' 프로모션 노출")
            st.write("- **굿즈 연계:** 선호 아티스트의 콘서트 티켓 우선 예매권 추첨 기회 제공")
        with col2:
            st.success("**🗣️ MGM (친구 추천)**")
            st.write("- **친구 초대 이벤트:** 친구 초대 시 양쪽 모두에게 1개월 무료 혜택 제공 (가장 강력한 마케팅 채널)")
    
    with tab4:
        st.markdown("#### ⚙️ 프로덕트 및 데이터 개선 방향")
        st.markdown("""
        > **데이터가 말해주는 서비스의 약점**
        
        1.  **'탐색 피로' 해결:** 곡을 1분도 안 듣고 넘기는 유저가 많음 → **'하이라이트 미리듣기'** 기능 도입 검토 필요
        2.  **광고 경험 개선:** 광고 도중 앱 종료율이 높음 → 청취 흐름을 끊지 않는 **'오디오 배너 광고'** 비중 확대
        3.  **초기 적응 실패:** 가입 첫 주 청취 시간이 20분 미만인 유저는 90% 이탈함 → **온보딩(Onboarding) 프로세스** 전면 개편 필요
        """)

def main():
    st.sidebar.markdown(
        """
        <style>
        .st-emotion-cache-16txtl3 {
            padding-top: 2rem;
        }
        .stRadio > label {
            font-weight: bold;
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }
        div[role="radiogroup"] label[data-baseweb="radio"] {
            background-color: transparent;
            padding: 10px;
            border-radius: 8px;
            transition: background-color 0.3s;
        }
         div[role="radiogroup"] label[data-baseweb="radio"]:hover {
            background-color: #f0f2f6;
        }
        div[role="radiogroup"] > label[aria-checked="true"] {
             background-color: #e6f7ed !important;
             color: #1DB954 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Navigation")
    
    menu_options = ["홈 (프로젝트 개요)", "모델 성능 비교", "실전 이탈 예측 & 심층 분석", "비즈니스 인사이트"]
    
    page = st.sidebar.radio("메뉴 이동", menu_options, label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    
    if page == "홈 (프로젝트 개요)":
        page_home()
    elif page == "모델 성능 비교":
        page_dashboard()
    elif page == "실전 이탈 예측 & 심층 분석":
        page_prediction()
    elif page == "비즈니스 인사이트":
        page_insights()

if __name__ == "__main__":
    main()