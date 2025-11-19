# import streamlit as st
# import pandas as pd
# import requests
# import joblib
# import numpy as np

# # --- Configuration ---
# # Assuming your FastAPI server is running on localhost:8000
# API_URL = "https://ml-model-bbdyddpgahpnr9amfz9odn.streamlit.app/predict" 
# # Assuming you saved the metrics in your training notebook
# METRICS = {
#     "Accuracy": 0.856, # Replace with your actual value from the TPOT output
#     "F1 Score": 0.655,   # Replace with your actual value
#     "AUC Score": 0.798    # Replace with your actual value
# }

# # --- Feature Importance Simulation ---
# # **NOTE**: TPOT's fitted pipeline might contain a non-tree-based classifier (like a K-Nearest Neighbor)
# # that doesn't natively expose a .feature_importances_ or .coef_ attribute.
# # To get the true importance/coefficients, you'd need to extract the final estimator from
# # the pipeline (best_model.steps[-1][1]) and check if it supports it.
# # For this example, we will use a set of dummy top 5 features.
# # You will need to calculate these based on your *actual* best_model.
# TOP_5_FEATURE_IMPORTANCE = {
#     "education_num": 0.35,
#     "net_capital": 0.25, # This is your engineered feature
#     "age": 0.15,
#     "hours_per_week": 0.10,
#     "marital_status_Married-civ-spouse": 0.05
# }

# # --- Data for Dropdowns (Based on the Adult Dataset) ---
# # Use unique values from the dataset, including NaN/missing values handled as '?' in the original data
# WORKCLASS_OPTIONS = ['Private', 'Self-emp-not-inc', 'Local-gov', '?', 'State-gov', 'Federal-gov', 'Without-pay', 'Self-emp-inc', 'Never-worked']
# EDUCATION_OPTIONS = ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', '11th', 'Assoc-acdm', '10th', '7th-8th', 'Prof-school', '9th', '12th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool']
# MARITAL_STATUS_OPTIONS = ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
# OCCUPATION_OPTIONS = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', '?', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
# RELATIONSHIP_OPTIONS = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
# RACE_OPTIONS = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
# SEX_OPTIONS = ['Male', 'Female']
# COUNTRY_OPTIONS = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'France', 'Greece', 'Ecuador', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Thailand', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-PR)', 'Hungary', 'Honduras', 'Scotland', 'Cambodia', 'Holand-Netherlands', '?']


# # --- Streamlit App Layout ---
# st.set_page_config(page_title="Adult Income Predictor", layout="wide")
# st.title("💰 Adult Income Predictor (FastAPI + Streamlit)")
# st.markdown("---")

# ## Model Performance Metrics
# st.header("📈 Model Test Performance (TPOT Classifier)")
# col1, col2, col3 = st.columns(3)
# col1.metric("Accuracy", f"{METRICS['Accuracy']:.3f}", delta=None)
# col2.metric("F1 Score", f"{METRICS['F1 Score']:.3f}", delta=None)
# col3.metric("AUC Score", f"{METRICS['AUC Score']:.3f}", delta=None)

# st.markdown("---")

# ## Feature Importance
# st.header("✨ Top 5 Feature Importance")
# importance_df = pd.DataFrame(TOP_5_FEATURE_IMPORTANCE.items(), columns=['Feature', 'Importance'])
# importance_df = importance_df.sort_values(by='Importance', ascending=False)
# st.bar_chart(importance_df.set_index('Feature'))

# st.markdown("---")

# ## Prediction Interface
# st.header("👤 Input Features for Income Prediction")

# # Create two columns for a better layout
# input_col_1, input_col_2 = st.columns(2)

# # Column 1 Inputs
# with input_col_1:
#     age = st.slider("Age", 17, 90, 30)
#     workclass = st.selectbox("Workclass", WORKCLASS_OPTIONS)
#     education = st.selectbox("Education", EDUCATION_OPTIONS)
#     education_num = st.slider("Education-Num (Years of Schooling)", 1, 16, 10)
#     marital_status = st.selectbox("Marital Status", MARITAL_STATUS_OPTIONS)
#     occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS)
#     relationship = st.selectbox("Relationship", RELATIONSHIP_OPTIONS)
    
# # Column 2 Inputs
# with input_col_2:
#     race = st.selectbox("Race", RACE_OPTIONS)
#     sex = st.selectbox("Sex", SEX_OPTIONS)
#     capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
#     capital_loss = st.number_input("Capital Loss", min_value=0, max_value=99999, value=0)
#     hours_per_week = st.slider("Hours per Week", 1, 99, 40)
#     native_country = st.selectbox("Native Country", COUNTRY_OPTIONS)


# # --- Prediction Button ---
# if st.button("Predict Income", type="primary"):
#     # 1. Create the request body based on the FastAPI Pydantic model
#     input_data = {
#         "age": age,
#         "workclass": workclass,
#         "education": education,
#         "education_num": education_num,
#         "marital_status": marital_status,
#         "occupation": occupation,
#         "relationship": relationship,
#         "race": race,
#         "sex": sex,
#         "capital_gain": capital_gain,
#         "capital_loss": capital_loss,
#         "hours_per_week": hours_per_week,
#         "native_country": native_country
#     }
    
#     try:
#         # 2. Make the POST request to the FastAPI endpoint
#         response = requests.post(API_URL, json=input_data)
        
#         # 3. Check for successful response
#         if response.status_code == 200:
#             result = response.json()
#             income_bracket = result["income_bracket"]
#             probabilities = result["probabilities"]
            
#             # 4. Display the prediction
#             st.success("### Prediction Result")
#             st.metric(label="Predicted Income Bracket", value=income_bracket)
#             st.info(f"Probability of <=50K: **{probabilities[0]*100:.2f}%**")
#             st.info(f"Probability of >50K: **{probabilities[1]*100:.2f}%**")
            
#         else:
#             st.error(f"Error calling the API. Status Code: {response.status_code}")
#             st.json(response.json())
            
#     except requests.exceptions.ConnectionError:
#         st.error("Connection Error: Could not connect to the FastAPI server. Please ensure the API is running at http://localhost:8000 (i.e., you ran `uvicorn api:app --reload` in your terminal).")
#     except Exception as e:
#         st.error(f"An unexpected error occurred: {e}")

# # --- How to Run ---
# st.markdown("---")
# st.subheader("How to Run This App")
# st.code("""
# # 1. Ensure your FastAPI backend is running:
# uvicorn api:app --reload

# # 2. Run this Streamlit frontend in a separate terminal:
# streamlit run frontend_app.py
# """)





import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuration & Model Loading (Adapted from your API) ---
# NOTE: Ensure 'preprocessor.joblib' and 'best_model.joblib' are present.
try:
    # 1. Load the preprocessor and the trained model
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('best_model.joblib')
    MODEL_LOADED = True
except FileNotFoundError:
    st.error("🚨 Model files ('preprocessor.joblib' and 'best_model.joblib') not found.")
    st.info("Please ensure the model files are in the same directory as this script.")
    MODEL_LOADED = False

# Remove API_URL since we are calling the prediction function locally
# API_URL = "..." 

# Assuming you saved the metrics in your training notebook
METRICS = {
    "Accuracy": 0.856, # Replace with your actual value from the TPOT output
    "F1 Score": 0.655,    # Replace with your actual value
    "AUC Score": 0.798     # Replace with your actual value
}

# --- Feature Importance Simulation ---
TOP_5_FEATURE_IMPORTANCE = {
    "education_num": 0.35,
    "net_capital": 0.25, # This is your engineered feature
    "age": 0.15,
    "hours_per_week": 0.10,
    "marital_status_Married-civ-spouse": 0.05
}

# --- Data for Dropdowns (Based on the Adult Dataset) ---
WORKCLASS_OPTIONS = ['Private', 'Self-emp-not-inc', 'Local-gov', '?', 'State-gov', 'Federal-gov', 'Without-pay', 'Self-emp-inc', 'Never-worked']
EDUCATION_OPTIONS = ['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc', '11th', 'Assoc-acdm', '10th', '7th-8th', 'Prof-school', '9th', '12th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool']
MARITAL_STATUS_OPTIONS = ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
OCCUPATION_OPTIONS = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', '?', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
RELATIONSHIP_OPTIONS = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
RACE_OPTIONS = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
SEX_OPTIONS = ['Male', 'Female']
COUNTRY_OPTIONS = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'France', 'Greece', 'Ecuador', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Thailand', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-PR)', 'Hungary', 'Honduras', 'Scotland', 'Cambodia', 'Holand-Netherlands', '?']

# --- Local Prediction Function (Adapted from your FastAPI logic) ---
def predict_income_local(input_data: dict):
    """Processes input data, makes a prediction using the loaded model, and returns the result."""
    
    # 1. Convert input dictionary data to a pandas DataFrame
    # Note: Use [input_data] to create a DataFrame with one row
    input_df = pd.DataFrame([input_data])

    # 2. Recreate the 'net_capital' feature exactly as done during training
    input_df['net_capital'] = input_df['capital_gain'] - input_df['capital_loss']

    # 3. Preprocess the input data
    processed_data = preprocessor.transform(input_df)

    # 4. Make prediction
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0].tolist()

    # 5. Return result dictionary
    return {
        "prediction": int(prediction),
        "income_bracket": ">50K" if prediction == 1 else "<=50K",
        "probabilities": probability
    }

# --- Streamlit App Layout ---
st.set_page_config(page_title="Adult Income Predictor", layout="wide")
st.title("💰 Adult Income Predictor (Local ML Model)")
st.markdown("---")

if MODEL_LOADED:
    ## Model Performance Metrics
    st.header("📈 Model Test Performance (TPOT Classifier)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{METRICS['Accuracy']:.3f}", delta=None)
    col2.metric("F1 Score", f"{METRICS['F1 Score']:.3f}", delta=None)
    col3.metric("AUC Score", f"{METRICS['AUC Score']:.3f}", delta=None)

    st.markdown("---")

    ## Feature Importance
    st.header("✨ Top 5 Feature Importance")
    importance_df = pd.DataFrame(TOP_5_FEATURE_IMPORTANCE.items(), columns=['Feature', 'Importance'])
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    st.bar_chart(importance_df.set_index('Feature'))

    st.markdown("---")

    ## Prediction Interface
    st.header("👤 Input Features for Income Prediction")

    # Create two columns for a better layout
    input_col_1, input_col_2 = st.columns(2)

    # Column 1 Inputs
    with input_col_1:
        age = st.slider("Age", 17, 90, 30)
        workclass = st.selectbox("Workclass", WORKCLASS_OPTIONS)
        education = st.selectbox("Education", EDUCATION_OPTIONS)
        education_num = st.slider("Education-Num (Years of Schooling)", 1, 16, 10)
        marital_status = st.selectbox("Marital Status", MARITAL_STATUS_OPTIONS)
        occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS)
        relationship = st.selectbox("Relationship", RELATIONSHIP_OPTIONS)
        
    # Column 2 Inputs
    with input_col_2:
        race = st.selectbox("Race", RACE_OPTIONS)
        sex = st.selectbox("Sex", SEX_OPTIONS)
        capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, max_value=99999, value=0)
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        native_country = st.selectbox("Native Country", COUNTRY_OPTIONS)


    # --- Prediction Button ---
    if st.button("Predict Income", type="primary"):
        # 1. Create the input data dictionary
        input_data = {
            "age": age,
            "workclass": workclass,
            "education": education,
            "education_num": education_num,
            "marital_status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital_gain": capital_gain,
            "capital_loss": capital_loss,
            "hours_per_week": hours_per_week,
            "native_country": native_country
        }
        
        try:
            # 2. Call the local prediction function directly
            result = predict_income_local(input_data)
            
            # 3. Display the prediction
            income_bracket = result["income_bracket"]
            probabilities = result["probabilities"]
            
            st.success("### Prediction Result")
            st.metric(label="Predicted Income Bracket", value=income_bracket)
            st.info(f"Probability of <=50K: **{probabilities[0]*100:.2f}%**")
            st.info(f"Probability of >50K: **{probabilities[1]*100:.2f}%**")
                
        except Exception as e:
            st.error(f"An unexpected error occurred during local prediction: {e}")

# --- How to Run ---
st.markdown("---")
st.subheader("How to Run This App")
st.code("""
# Ensure you have the required libraries installed:
pip install streamlit pandas numpy joblib

# Run this Streamlit frontend in your terminal:
streamlit run your_app_file_name.py
""")