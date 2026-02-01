import streamlit as st
import joblib
import pandas as pd
import json
import requests
from datetime import datetime
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Weather2Go - Know the road before you go!",
    page_icon="⛅🚗",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8f6 50%, #e3f2fd 100%);
    }
    
    /* Adjust padding */
    div.block-container {
        padding-top: 1rem;
    }
    
    /* Title styling */
    h1 {
        color: #1b5e20;
        margin-top: -2rem !important;
        text-align: center;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #2e7d32;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #1b5e20;
    }
    
    /* Card backgrounds */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #c8e6c9 0%, #b2dfdb 100%);
    }
    
    /* Button styling */
    .stButton button {
        background-color: #43a047;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton button:hover {
        background-color: #2e7d32;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Info/warning/error boxes */
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Header logo (place your logo at assets/logo.png)
logo_path = Path("assets/Weather2Go logo1.png")
if logo_path.exists():
    st.markdown("<style>div.block-container{padding-top:1rem;}</style>", unsafe_allow_html=True)
    left, center, right = st.columns([0.8, 1, 0.55])
    with center:
        st.image(str(logo_path), width=310)



# Load model artifacts
@st.cache_resource
def load_model():
    model = joblib.load('model_artifacts/rf_model.joblib')
    label_encoders = joblib.load('model_artifacts/label_encoders.joblib')
    with open('model_artifacts/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    return model, label_encoders, metadata

# Fetch weather data from NOAA API (no key needed!)
def get_weather_data(city):
    """Fetch current weather data using NOAA API"""
    
    # Map Michigan cities to coordinates
    michigan_cities = {
        'detroit': (42.3314, -83.0458),
        'ann arbor': (42.2808, -83.7430),
        'grand rapids': (42.9633, -85.6789),
        'lansing': (42.7335, -84.5555),
        'flint': (43.0125, -83.6875),
        'dearborn': (42.3222, -83.1763),
        'sterling heights': (42.5800, -83.0300),
        'troy': (42.5801, -83.1481),
        'warren': (42.4805, -83.0267),
        'livonia': (42.4172, -83.3711),
        'kalamazoo': (42.2917, -85.5872),
        'saginaw': (43.4209, -83.9545),
        'muskegon': (43.2343, -86.2474),
        'jackson': (42.2411, -84.4054),
        'battle creek': (42.3202, -85.1832),
    }
    
    city_lower = city.lower().strip()
    
    if city_lower not in michigan_cities:
        st.error(f"City '{city}' not found in database. Available cities:")
        available = ", ".join([c.title() for c in sorted(michigan_cities.keys())])
        st.write(available)
        return None
    
    lat, lon = michigan_cities[city_lower]
    
    try:
        # Step 1: Get grid point data
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        points_response = requests.get(points_url, timeout=10)
        
        if points_response.status_code != 200:
            st.error(f"Error getting location data: {points_response.status_code}")
            return None
        
        points_data = points_response.json()
        forecast_url = points_data['properties']['forecast']
        
        # Step 2: Get forecast data
        forecast_response = requests.get(forecast_url, timeout=10)
        
        if forecast_response.status_code != 200:
            st.error(f"Error getting forecast data: {forecast_response.status_code}")
            return None
        
        forecast_data = forecast_response.json()
        
        # Get current period (usually the first one)
        current = forecast_data['properties']['periods'][0]
        
        # Extract weather data
        temp = float(current['temperature'])
        wind_speed = float(current['windSpeed'].split()[0])  # Extract number from "15 mph" format
        
        # Parse wind direction
        wind_dir_str = current['windDirection']  # N, NE, E, SE, S, SW, W, NW
        wind_direction = wind_dir_str if wind_dir_str != 'Calm' else 'N'
        
        # Determine day/night
        time_of_day = "Day" if current['isDaytime'] else "Night"
        
        # Get more detailed data from weather.gov grid (optional - has sensible defaults)
        humidity = 50
        visibility = 10.0
        precipitation = 0.0
        
        try:
            if 'gridpoints' in points_data['properties']:
                grid_url = points_data['properties']['gridpoints'].replace('/wfo/', '/').replace('/grid/', '/')
                grid_response = requests.get(grid_url, timeout=10)
                
                if grid_response.status_code == 200:
                    grid_data = grid_response.json()
                    props = grid_data['properties']
                    
                    # Extract values from the first element of each array
                    if props.get('relativeHumidity', {}).get('values'):
                        humidity = float(props['relativeHumidity']['values'][0]['value']) if props['relativeHumidity']['values'][0].get('value') else 50
                    
                    # Get visibility (in meters)
                    if props.get('visibility', {}).get('values'):
                        visibility_m = float(props['visibility']['values'][0]['value']) if props['visibility']['values'][0].get('value') else 10000
                        visibility = visibility_m / 1609.34  # Convert to miles
                    
                    # Get quantitative precipitation
                    if props.get('quantitativePrecipitation', {}).get('values'):
                        qpf = float(props['quantitativePrecipitation']['values'][0]['value']) if props['quantitativePrecipitation']['values'][0].get('value') else 0
                        precipitation = qpf / 25.4 if qpf else 0  # Convert mm to inches
        except Exception as e:
            # Use default values if grid data fails
            pass
        
        # Calculate wind chill
        if temp <= 50 and wind_speed > 3:
            wind_chill = 35.74 + (0.6215 * temp) - (35.75 * (wind_speed ** 0.16)) + (0.4275 * temp * (wind_speed ** 0.16))
        else:
            wind_chill = temp
        
        # Estimate pressure (average for Michigan)
        pressure = 29.9
        
        return {
            'City': city,
            'Temperature(F)': temp,
            'Wind_Chill(F)': wind_chill,
            'Humidity(%)': humidity,
            'Pressure(in)': pressure,
            'Visibility(mi)': visibility,
            'Wind_Direction': wind_direction,
            'Wind_Speed(mph)': wind_speed,
            'Precipitation(in)': precipitation,
            'Sunrise_Sunset': time_of_day
        }
        
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please check your internet connection and try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🌐 Connection error. Please check your internet connection.")
        return None
    except KeyError as e:
        st.error(f"Data parsing error: Missing field {e}")
        return None
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None



try:
    model, label_encoders, metadata = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False


# App title and description
st.markdown("<style>h1{margin-top:-2rem !important;}</style>", unsafe_allow_html=True)
st.title("Weather2Go - Know the Road Before You Go")
st.markdown("""
Predict accident risk levels (Low, Medium, High) based on current weather conditions in Michigan.
Simply enter your destination city and we'll analyze the current weather to get your driving risk assessment.
""")

if model_loaded:
    # Simple city input
    st.subheader("📍 Enter Your Destination")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        city = st.text_input(
            "City in Michigan",
            value="Detroit",
            placeholder="e.g., Detroit, Ann Arbor, Grand Rapids",
            help="Enter any city in Michigan"
        )
    
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        predict_button = st.button("🔮 Check Risk Level", type="primary", use_container_width=True)
    
    # Predict button
    if predict_button:
        if not city.strip():
            st.error("Please enter a city name.")
        else:
            with st.spinner(f"🌤️ Fetching current weather data for {city}..."):
                weather_data = get_weather_data(city)
            
            if weather_data:
                # Display weather data
                st.markdown("---")
                st.subheader("🌤️ Current Weather Conditions")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Temperature", f"{weather_data['Temperature(F)']:.1f}°F")
                    st.metric("Wind Chill", f"{weather_data['Wind_Chill(F)']:.1f}°F")
                
                with col2:
                    st.metric("Humidity", f"{weather_data['Humidity(%)']:.0f}%")
                    st.metric("Pressure", f"{weather_data['Pressure(in)']:.2f} in")
                
                with col3:
                    st.metric("Visibility", f"{weather_data['Visibility(mi)']:.1f} mi")
                    st.metric("Precipitation", f"{weather_data['Precipitation(in)']:.2f} in")
                
                with col4:
                    st.metric("Wind Speed", f"{weather_data['Wind_Speed(mph)']:.1f} mph")
                    st.metric("Wind Direction", weather_data['Wind_Direction'])
                
                st.info(f"⏰ Time of Day: **{weather_data['Sunrise_Sunset']}**")
                
                try:
                    # Create DataFrame
                    df = pd.DataFrame([weather_data])
                    
                    # Encode categorical columns
                    for col, encoder in label_encoders.items():
                        if col in df.columns:
                            try:
                                df[col] = encoder.transform(df[col].astype(str))
                            except ValueError:
                                # Handle unknown categories
                                st.warning(f"Unknown value for {col}. Using most common value.")
                                df[col] = 0
                    
                    # Make prediction
                    prediction = model.predict(df)[0]
                    probabilities = model.predict_proba(df)[0]
                    
                    # Get class names
                    class_names = model.classes_
                    prob_dict = dict(zip(class_names, probabilities))
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Risk Assessment")
                    
                    # Risk level with color coding
                    risk_colors = {
                        'Low': '🟢',
                        'Medium': '🟡',
                        'High': '🔴'
                    }
                    
                    st.markdown(f"## {risk_colors.get(prediction, '⚪')} Risk Level: **{prediction}**")
                    
                    # Probability bars
                    st.markdown("### Confidence Levels:")
                    for risk_level in ['High', 'Medium', 'Low']:
                        if risk_level in prob_dict:
                            prob = prob_dict[risk_level]
                            st.progress(prob, text=f"{risk_level}: {prob*100:.1f}%")
                    
                    # Risk interpretation
                    st.markdown("---")
                    st.subheader("💡 Driving Recommendations")
                    
                    if prediction == 'High':
                        st.error("""
                        **⚠️ High Risk Detected**
                        - Consider delaying travel if possible
                        - Drive with extreme caution
                        - Increase following distance significantly
                        - Reduce speed by 20-30%
                        - Ensure all safety equipment is functional
                        - Stay updated on weather conditions
                        """)
                    elif prediction == 'Medium':
                        st.warning("""
                        **⚠️ Moderate Risk Detected**
                        - Exercise increased caution
                        - Adjust speed for conditions (reduce by 10-15%)
                        - Increase following distance
                        - Stay alert for hazards
                        - Consider alternative routes if available
                        """)
                    else:
                        st.success("""
                        **✅ Low Risk Detected**
                        - Conditions are relatively safe for driving
                        - Standard safe driving practices apply
                        - Stay aware of your surroundings
                        - Continue monitoring weather conditions
                        - Safe driving helps avoid claims that could affect your insurance rate or score
                        """)

                    st.markdown("""
                    ### 🛡️ Insurance Risk & Costs
                    - 💸 Accidents can raise premiums due to higher perceived risk
                    - 🧾 Claims may trigger deductibles, reach coverage limits, or affect renewal
                    - 📉 A cleaner driving record helps keep insurance more affordable
                    """)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.info("Please try again or check if the model artifacts are properly loaded.")

else:
    st.error("⚠️ Model could not be loaded. Please ensure model artifacts exist in 'model_artifacts/' folder.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About Weather2Go")
    st.markdown("""
    **Weather2Go** uses a Random Forest machine learning model trained on historical
    Michigan accident data to predict risk levels based on real-time weather conditions.
    
    ### How It Works:
    1. Enter your destination city
    2. We fetch live weather data
    3. Model predicts accident risk level
    4. Get personalized safety recommendations
    
    ### Weather Factors Analyzed:
    - 🌡️ Temperature & Wind Chill
    - 💨 Wind Speed & Direction
    - 💧 Humidity & Precipitation
    - 👁️ Visibility
    - 🌅 Time of Day (Day/Night)
    
    ### Risk Levels:
    - 🟢 **Low**: Safe driving conditions
    - 🟡 **Medium**: Exercise caution
    - 🔴 **High**: Hazardous conditions
    
    ---
    
    ### 🔑 API Setup
    This app uses the **NOAA Weather API** which doesn't require an API key!
    
    Weather data is fetched directly from the National Weather Service.
    
    **Supported Michigan Cities:**
    - Detroit, Ann Arbor, Grand Rapids, Lansing, Flint
    - Dearborn, Sterling Heights, Troy, Warren, Livonia
    - Kalamazoo, Saginaw, Muskegon, Jackson, Battle Creek
    
    ---
    *Built for SpartaHack 11*
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    if model_loaded:
        st.info(f"""
        - Model: Random Forest
        - Features: {len(metadata['feature_names'])}
        - Classes: {', '.join(metadata['classes'])}
        - Data Source: Michigan Accidents
        """)
    
    # API Status indicator
    st.markdown("---")
    st.success("✅ Weather API: NOAA (No API key needed)")
