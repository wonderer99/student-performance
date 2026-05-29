import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go


st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    .result-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .result-score {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)


class perform(nn.Module):
    def __init__(self):
        super(perform, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_model():
    model = perform()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    scaler_X = joblib.load("scaler_X.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_model()


st.title("🎓 Student Performance Predictor")
st.markdown("Fill in the student details below and click **Predict** to estimate their performance score.")
st.divider()


col1, col2 = st.columns(2)

with col1:
    hours_studied = st.slider("📚 Hours Studied per Day", min_value=1, max_value=9, value=5, step=1)
    previous_scores = st.slider("📝 Previous Scores (%)", min_value=40, max_value=99, value=70, step=1)
    sleep_hours = st.slider("😴 Sleep Hours per Night", min_value=4, max_value=9, value=7, step=1)

with col2:
    sample_papers = st.number_input("📄 Sample Question Papers Practiced", min_value=0, max_value=9, value=3, step=1)
    extracurricular_label = st.radio("⚽ Extracurricular Activities", options=["Yes", "No"], horizontal=True)
    extracurricular = 1 if extracurricular_label == "Yes" else 0


st.divider()


if st.button("🔮 Predict Performance", use_container_width=True, type="primary"):

    input_data = np.array([[hours_studied, previous_scores, extracurricular,
                            sleep_hours, sample_papers]], dtype=np.float32)
    input_scaled = scaler_X.transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled)

    with torch.no_grad():
        prediction_scaled = model(input_tensor)

    prediction_real = scaler_y.inverse_transform(prediction_scaled.numpy())
    score = float(prediction_real[0][0])
    score = max(0, min(100, score))

    st.markdown(f"""
        <div class="result-card">
            <p style="color: #a0aec0; font-size: 1rem; margin-bottom: 0.5rem;">Predicted Performance Index</p>
            <div class="result-score">{score:.1f}</div>
            <p style="color: #718096; font-size: 0.85rem; margin-top: 0.5rem;">out of 100</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Your Score", f"{score:.1f}")
    col_b.metric("vs. Average", "55.0", f"{score - 55:.1f}")
    col_c.metric("Grade",
                 "A+" if score >= 90 else
                 "A"  if score >= 80 else
                 "B"  if score >= 70 else
                 "C"  if score >= 60 else "D")

    if score >= 80:
        st.success("🌟 Excellent performance! Keep it up.")
    elif score >= 60:
        st.warning("📈 Good, but there's room to improve!")
    else:
        st.error("💪 Needs more effort — try studying more hours!")


with st.expander("📊 What affects Performance the most?", expanded=True):

    @st.cache_data
    def load_correlation():
        df = pd.read_csv("Student_Performance.csv")
        df.columns = df.columns.str.lower().str.strip()
        df['extracurricular activities'] = df['extracurricular activities'].map({'Yes': 1, 'No': 0})
        corr = df.corr()
        target_corr = corr['performance index'].drop('performance index')
        target_corr = target_corr.reindex(target_corr.abs().sort_values(ascending=True).index)
        return target_corr

    target_corr = load_correlation()

    bar_colors = ['#f87171' if v < 0 else '#34d399' for v in target_corr.values]

    fig = go.Figure(go.Bar(
        x=target_corr.values,
        y=target_corr.index,
        orientation='h',
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in target_corr.values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>'
    ))

    fig.update_layout(
        title={'text': "Correlation with Performance Index", 'font': {'size': 16, 'color': '#e2e8f0'}},
        xaxis={
        'title': {
            'text': 'Your X Axis Label',
            'font': {'size': 14, 'color': '#e2e8f0'}  # This replaces titlefont
        }
    },
        yaxis={
        'title': {
            'text': 'Your Y Axis Label',
            'font': {'size': 14, 'color': '#e2e8f0'}  # This replaces titlefont
        }
    },
    height=320,
)

    st.plotly_chart(fig, use_container_width=True)

    col_leg1, col_leg2 = st.columns(2)
    col_leg1.markdown("🟢 **Positive** — higher value → better score")
    col_leg2.markdown("🔴 **Negative** — higher value → lower score")

st.divider()


with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a **PyTorch neural network** trained on student data
    to predict academic performance.

    **Model Architecture:**
    - Input: 5 features
    - Hidden: 64 → 32 neurons (ReLU)
    - Output: 1 score

    **Features used:**
    1. Hours Studied
    2. Previous Scores
    3. Extracurricular Activities
    4. Sleep Hours
    5. Sample Papers Practiced
    """)
    st.divider()
    st.caption("Built with Streamlit + PyTorch")