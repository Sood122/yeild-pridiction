import streamlit as st
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Page config
st.set_page_config(page_title="Crop Recommendation System", layout="centered")

st.title("🌾 Crop Yield & Recommendation System using Fuzzy Logic & Season")

# --- Optional: Load crop data from GitHub ---
@st.cache_data
def load_crop_data():
    url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO_NAME/master/crop_data.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not load crop data: {e}")
        return None

crop_df = load_crop_data()

# Input sliders
rain = st.slider("🌧️ Rainfall (mm)", 0, 200, 100)
temp = st.slider("🌡️ Temperature (°C)", 10, 50, 30)
fert = st.slider("💊 Fertilizer (kg/acre)", 0, 200, 100)

season = st.selectbox("📅 Select Season", ["Kharif", "Rabi", "Zaid"])

# Fuzzy system setup
rainfall = ctrl.Antecedent(np.arange(0, 201, 1), 'rainfall')
temperature = ctrl.Antecedent(np.arange(10, 51, 1), 'temperature')
fertilizer = ctrl.Antecedent(np.arange(0, 201, 1), 'fertilizer')
recommendation = ctrl.Consequent(np.arange(0, 11, 1), 'recommendation')

rainfall['low'] = fuzz.trimf(rainfall.universe, [0, 0, 100])
rainfall['medium'] = fuzz.trimf(rainfall.universe, [50, 100, 150])
rainfall['high'] = fuzz.trimf(rainfall.universe, [100, 200, 200])

temperature['low'] = fuzz.trimf(temperature.universe, [10, 10, 25])
temperature['medium'] = fuzz.trimf(temperature.universe, [20, 30, 40])
temperature['high'] = fuzz.trimf(temperature.universe, [35, 50, 50])

fertilizer['low'] = fuzz.trimf(fertilizer.universe, [0, 0, 100])
fertilizer['medium'] = fuzz.trimf(fertilizer.universe, [50, 100, 150])
fertilizer['high'] = fuzz.trimf(fertilizer.universe, [100, 200, 200])

recommendation['poor'] = fuzz.trimf(recommendation.universe, [0, 0, 5])
recommendation['average'] = fuzz.trimf(recommendation.universe, [3, 5, 8])
recommendation['good'] = fuzz.trimf(recommendation.universe, [6, 10, 10])

# Fuzzy rules
rules = [
    ctrl.Rule(rainfall['low'] & temperature['low'], recommendation['poor']),
    ctrl.Rule(rainfall['medium'] & temperature['medium'] & fertilizer['medium'], recommendation['good']),
    ctrl.Rule(rainfall['high'] & temperature['high'], recommendation['poor']),
    ctrl.Rule(rainfall['medium'] & fertilizer['high'], recommendation['average']),
]

recommendation_ctrl = ctrl.ControlSystem(rules)
recommendation_sim = ctrl.ControlSystemSimulation(recommendation_ctrl)

# Apply inputs
recommendation_sim.input['rainfall'] = rain
recommendation_sim.input['temperature'] = temp
recommendation_sim.input['fertilizer'] = fert
recommendation_sim.compute()

# Output
st.subheader("📈 Recommendation Score (0-10):")
st.success(f"{recommendation_sim.output['recommendation']:.2f}")

# Season-based crop suggestions
season_crop_map = {
    'Kharif': ['Rice', 'Maize', 'Cotton'],
    'Rabi': ['Wheat', 'Mustard', 'Barley'],
    'Zaid': ['Watermelon', 'Cucumber', 'Moong']
}
recommended_crops = season_crop_map.get(season, [])

st.subheader(f"🌿 Best Crops for {season} Season:")
for crop in recommended_crops:
    st.markdown(f"- {crop}")

# Optional: Show crop dataset preview
if crop_df is not None:
    st.subheader("📊 Uploaded Crop Dataset Preview")
    st.dataframe(crop_df.head())
