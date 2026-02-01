import streamlit as st
import joblib
import requests
import pandas as pd
from datetime import datetime

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('model_artifacts/rf_model.joblib')

# Weather API function
def get_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'  # For Celsius
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None

# Main app
def main():
    st.title("Weather2Go")

    # Load model
    model = load_model()

    # Get API key from secrets
    weather_api_key = st.secrets["weather_api_key"]

    # User input form
    with st.form("user_input"):
        # Personal info
        name = st.text_input("Name")
        city1 = st.text_input("City1")
        city2 = st.text_input("City2")

        # Model input features
        feature1 = st.number_input("Feature 1", min_value=0.0)
        feature2 = st.number_input("Feature 2", min_value=0.0)
        # Add more features as needed

        submitted = st.form_submit_button("Submit")

    if submitted:
        # Get current weather
        weather_data = get_weather(weather_api_key, city)

        if weather_data:
            # Extract relevant weather info
            temperature = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            weather_condition = weather_data['weather'][0]['main']

            # Prepare model input
            model_input = pd.DataFrame({
                'feature1': [feature1],
                'feature2': [feature2],
                'temperature': [temperature],
                'humidity': [humidity]
                # Add more features as needed
            })

            # Make prediction
            risk_score = model.predict(model_input)[0]

            # Display results
            st.subheader("Results")

            col1, col2 = st.columns(2)

            with col1:
                st.write("Weather Conditions:")
                st.write(f"- Temperature: {temperature}°C")
                st.write(f"- Humidity: {humidity}%")
                st.write(f"- Condition: {weather_condition}")

            with col2:
                st.write("Risk Assessment:")
                st.write(f"- Risk Score: {risk_score:.2f}")

                # Custom risk levels based on score
                if risk_score < 0.3:
                    risk_level = "Low"
                    color = "green"
                elif risk_score < 0.7:
                    risk_level = "Medium"
                    color = "yellow"
                else:
                    risk_level = "High"
                    color = "red"

                st.markdown(f"- Risk Level: :{color}[{risk_level}]")

            # Save to database/file
            save_data = {
                'timestamp': datetime.now(),
                'name': name,
                'city1': city1,
                'city1': city2,
                'weather_condition': weather_condition,
                'temperature': temperature,
                'humidity': humidity,
                'risk_score': risk_score,
                'risk_level': risk_level
            }

            # Use Streamlit connection to save to database
            conn = st.connection('results_db', type='sql')
            with conn.session as s:
                s.execute(
                    """INSERT INTO results 
                       (timestamp, name, city, weather_condition, temperature, 
                        humidity, risk_score, risk_level)
                       VALUES (:ts, :name, :city, :weather, :temp, :humid, 
                              :score, :level)""",
                    params=save_data
                )
                s.commit()

if __name__ == "__main__":
    main()