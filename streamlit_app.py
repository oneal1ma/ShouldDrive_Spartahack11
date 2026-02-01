import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def init_model():
    """Initialize or load the safety prediction model"""
    try:
        # Try to load existing model
        model = joblib.load('rf_model.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        st.success("Loaded existing model")

        # Create and fit label encoder for weather conditions
        label_encoder = LabelEncoder()
        sample_data['weather_encoded'] = label_encoder.fit_transform(
            sample_data['weather_condition']
        )

        # Create and train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(
            sample_data[['temperature', 'weather_encoded', 'wind_speed', 'visibility']], 
            sample_data['safety_score']
        )

        # Save model and encoder
        joblib.dump(model, 'safety_model.joblib')
        joblib.dump(label_encoder, 'label_encoder.joblib')
        st.success("Created and saved new model")

    return model, label_encoder

def get_weather(city, api_key):
    """Fetch weather data from OpenWeather API"""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def predict_safety_score(weather_data, model, label_encoder):
    """Predict safety score using the ML model"""
    if not weather_data:
        return None, "Unable to fetch weather data"

    # Extract features from weather data
    features = pd.DataFrame({
        'temperature': [weather_data['main']['temp']],
        'weather_condition': [weather_data['weather'][0]['main']],
        'wind_speed': [weather_data['wind']['speed']],
        'visibility': [weather_data.get('visibility', 10000)]
    })

    # Encode weather condition
    features['weather_encoded'] = label_encoder.transform(features['weather_condition'])

    # Make prediction
    prediction = model.predict(
        features[['temperature', 'weather_encoded', 'wind_speed', 'visibility']]
    )[0]

    # Generate message
    messages = {
        1: "Safe driving conditions",
        2: "Moderate risk - Drive with caution",
        3: "High risk - Consider postponing travel"
    }

    return prediction, messages[prediction]

def main():
    st.title("🚗 ML-Powered Driving Safety Analyzer")

    # Initialize model
    model, label_encoder = init_model()

    # Get OpenWeather API key from secrets
    api_key = st.secrets["openweather"]["api_key"]

    # User input
    col1, col2 = st.columns(2)
    with col1:
        start_location = st.text_input("Starting City")
    with col2:
        end_location = st.text_input("Destination City")

    if st.button("Analyze Route Safety"):
        if start_location and end_location:
            # Get weather for both locations
            start_weather = get_weather(start_location, api_key)
            end_weather = get_weather(end_location, api_key)

            if start_weather and end_weather:
                # Calculate safety scores using ML model
                start_score, start_message = predict_safety_score(
                    start_weather, model, label_encoder
                )
                end_score, end_message = predict_safety_score(
                    end_weather, model, label_encoder
                )

                # Use worst score between start and end
                final_score = max(start_score, end_score)

                # Display results
                st.header("Safety Analysis")

                # Weather information
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"🏁 {start_location}")
                    st.write(f"Temperature: {start_weather['main']['temp']}°C")
                    st.write(f"Conditions: {start_weather['weather'][0]['description']}")
                    st.write(f"Wind Speed: {start_weather['wind']['speed']} m/s")
                    st.write(f"Visibility: {start_weather.get('visibility', 10000)} m")

                with col2:
                    st.subheader(f"🏁 {end_location}")
                    st.write(f"Temperature: {end_weather['main']['temp']}°C")
                    st.write(f"Conditions: {end_weather['weather'][0]['description']}")
                    st.write(f"Wind Speed: {end_weather['wind']['speed']} m/s")
                    st.write(f"Visibility: {end_weather.get('visibility', 10000)} m")

                # Safety Score
                st.subheader("Overall Safety Score")
                if final_score == 1:
                    st.success("✅ Safe driving conditions")
                elif final_score == 2:
                    st.warning("⚠️ Moderate risk - Drive with caution")
                else:
                    st.error("🚫 High risk - Consider postponing travel")

                # Model confidence
                predictions_proba = model.predict_proba(
                    [[start_weather['main']['temp'], 
                      label_encoder.transform([start_weather['weather'][0]['main']])[0],
                      start_weather['wind']['speed'],
                      start_weather.get('visibility', 10000)]]
                )
                st.write("Model Confidence:")
                st.write(pd.DataFrame(
                    predictions_proba,
                    columns=['Safe', 'Moderate Risk', 'High Risk']
                ))

            else:
                st.error("Unable to fetch weather data. Please check city names.")
        else:
            st.warning("Please enter both starting and destination cities.")

if __name__ == "__main__":
    main()