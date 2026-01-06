import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction App")
st.write("Predict house prices using trained model")

# Safe model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "house_price_model.pkl")

# Load model
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Inputs
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
sqft = st.number_input("Square Feet", 500, 10000, 2000)
lot_size = st.number_input("Lot Size", 1000, 20000, 8000)
age = st.number_input("House Age", 0, 100, 15)
year_built = st.number_input("Year Built", 1900, 2025, 2010)
garage = st.number_input("Garage Spaces", 0, 5, 2)

location = st.selectbox("Location", ["Downtown", "Suburban", "Rural"])
house_type = st.selectbox("House Type", ["House", "Apartment", "Villa"])
condition = st.slider("Condition (1–5)", 1, 5, 4)
school_rating = st.slider("School Rating (1–10)", 1, 10, 8)

has_pool = st.selectbox("Swimming Pool", [0, 1])
has_fireplace = st.selectbox("Fireplace", [0, 1])
has_basement = st.selectbox("Basement", [0, 1])

if st.button("Predict"):
    input_data = np.array([[bedrooms, bathrooms, sqft, lot_size, age,
                            year_built, garage, location, house_type,
                            condition, school_rating, has_pool, has_fireplace,
                            has_basement]])
    prediction = model.predict(input_data)
    st.success(f"🏷 Estimated Price: ₹{int(prediction[0]):,}")
