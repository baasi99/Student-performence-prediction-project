import streamlit as st # type: ignore
import pandas as pd # type: ignore
import joblib 
from PIL import Image  # type: ignore # For adding clip arts

# Load pre-trained model, scaler, and encoder
model = joblib.load('LinearRegression.pkl')
scaler = joblib.load('Scaler.pkl')
encoder = joblib.load('Encoder.pkl')
st.set_page_config(page_title="Background Image Example", layout="wide")

# Define the background image URL
background_image_url = "https://img.freepik.com/free-photo/girl-playing-game-instead-studying_23-2147833834.jpg?t=st=1742550465~exp=1742554065~hmac=dec574d4081d295b063d073be0f49978d920ccfe9c84517ced5423006d762c18&w=1380"
# Inject custom CSS for the background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Set page configuration
#st.set_page_config(page_title="EduPredict", layout="centered", initial_sidebar_state="auto")

# Create a blank image with white background

# Display header and clip art
st.markdown(
    "<h1 style='text-align: center; color: yellow; font-family: Arial, sans-serif; text-shadow: 0 0 10px rgba(255, 255, 0, 0.8), -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>EduPredict</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h5 style='text-align: center; color: white; font-family: Verdana, Geneva, sans-serif; text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);'>Forecast Your Academic Success Through Prediction</h5>",
    unsafe_allow_html=True
)
#st.write("-")  # Divider for a cleaner look

# Input section

st.markdown(
    """
    <style>
    /* Style input fields */
    # input[type=number] {
    #     border: 2px solid #f39c12;
    #     border-radius: 10px; /* Rounded corners */
    #     padding: 5px;
    #     # width: 100px; /* Shorter width */
    #     font-size: 14px;
    # }
    div[data-testid="stNumberInput"] {
        margin-top: 5px; /* Adjust space between input box and label */
        margin-bottom: 5px;
        width: 500%;
    }
    input[type=number]:focus {
        outline: none;
        box-shadow: 0px 0px 5px #f39c12; /* Glow effect */
    }

    /* Style the select box */
    select {
        border: 2px solid #f39c12;
        border-radius: 10px;
        padding: 5px;
        font-size: 14px;
    }
    /* Style form section */
    .form-section {
        background-color: #f9f9f9;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Page layout


# Sidebar with inputs

col1,col2 = st.columns(2)

with col1:
    hour_studied = st.number_input("📖 Hours Studied per Day", min_value=0, max_value=24, value=5)
    prev_score = st.number_input("📊 Past Academic Score", min_value=0, max_value=100, value=50)
    sleep_hours = st.number_input("😴 Sleep Hours per Day", min_value=0, max_value=24, value=7)
with col2:
    paper = st.number_input("📑 Sample Question Papers Practiced", min_value=0, value=2)
    eca = st.selectbox("🎭 Extracurricular Activities", ["Yes", "No"])

# Add custom CSS for opaque background
st.markdown(
    """
    <style>
    .stApp {
        background-color: rgba(255, 255, 255, 0.8); /* White background with 80% opacity */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Encode categorical data
eca_encoded = encoder.transform([eca])[0]  # Ensure encoder compatibility

# Create DataFrame for the input
data = pd.DataFrame({
    'Hours Studied': [hour_studied],
    'Previous Scores': [prev_score],
    'Sleep Hours': [sleep_hours],
    'Sample Question Papers Practiced': [paper],
    'ECA': [eca_encoded]
})

# Scale the data
data_scaled = scaler.transform(data)

# Prediction button and result display
# Prediction button
if st.button("🎯Predict", key="predict_button"):
    prediction = model.predict(data_scaled).item()  # Extracting single value
    st.success(f"📌 Predicted Student Performance is: **{prediction:.2f}**")

# Add custom CSS for the red button
st.markdown(
    """
    <style>
    div.stButton > button#predict_button {
        background-color: red;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
        cursor: pointer;
    }
    div.stButton > button#predict_button:hover {
        background-color: darkred;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Footer section for a professional look
st.write("---")
st.markdown(
    "<p style='text-align: center; color: white;'>EduPredict © 2025. All rights reserved.</p>",
    unsafe_allow_html=True
)
