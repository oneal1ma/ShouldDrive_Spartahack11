# Weather2Go 

Weather2Go is an accident risk prediction system that helps drivers make informed decisions about road safety in Michigan. Using machine learning and real-time weather data from the NOAA Weather API, the Streamlit web app predicts accident risk levels (Low, Medium, High) to promote safer driving conditions.

**Local App Link:** http://localhost:8505/

> Note: This link works only while the Streamlit app is running on your machine.

## Streamlit App

### Run the app locally
1. Install dependencies (see Installation below)
2. Start the app:
    ```bash
    streamlit run app.py
    ```

## Project Structure

### Machine Learning Model
- **Notebook**: `michigan-accidents-risk-prediction.ipynb`
- **Algorithm**: Random Forest Classifier with hyperparameter tuning
- **Features**: Weather conditions (temperature, humidity, wind, visibility, precipitation), location, and time of day
- **Performance**: Optimized using GridSearchCV with F1-weighted scoring

### Model Artifacts
The trained model and associated files are stored in the `model_artifacts/` directory:
- `rf_model.joblib` - Trained Random Forest model
- `label_encoders.joblib` - Encoders for categorical features (City, Wind Direction, Sunrise/Sunset)
- `model_metadata.json` - Model configuration and feature information

## Getting Started

### To Execute
1. Install the required libraries:
    ```bash
    pip install streamlit pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib requests
    ```
2. Clone or download the project.
3. Open the Jupyter notebook or script:
    - Notebook: `michigan-accidents-risk-prediction.ipynb`
    - App: `app.py`
4. Run the notebook cells to preprocess data, train the model, and export artifacts.
5. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

### Using the Trained Model
- Load model artifacts from `model_artifacts/`
- Encode categorical features (City, Wind Direction, Sunrise/Sunset)
- Run predictions and view risk probabilities

## Model Details
- **Features**: 10 weather/location inputs
- **Balancing**: SMOTE + undersampling
- **Split**: 80/20 stratified
- **Algorithm**: Random Forest (GridSearchCV, F1-weighted)
- **Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix, Feature Importance

