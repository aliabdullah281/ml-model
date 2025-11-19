import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# 1. Load the preprocessor and the trained model
try:
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('best_model.joblib')
except FileNotFoundError:
    raise Exception("Model files not found. Please run the notebook first to generate 'preprocessor.joblib' and 'best_model.joblib'.")

app = FastAPI()

# 2. Define the input data structure using Pydantic
class PredictRequest(BaseModel):
    age: int
    workclass: str
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

# 3. Define the prediction endpoint
@app.post("/predict")
def predict_income(data: PredictRequest):
    # Convert Pydantic model data to a pandas DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Recreate the 'net_capital' feature exactly as done during training
    input_df['net_capital'] = input_df['capital_gain'] - input_df['capital_loss']

    # Preprocess the input data
    processed_data = preprocessor.transform(input_df)

    # Make prediction
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0].tolist()

    # Return result
    return {
        "prediction": int(prediction),
        "income_bracket": ">50K" if prediction == 1 else "<=50K",
        "probabilities": probability
    }

# To run the API, save this file and run the following command in your terminal:
# uvicorn api:app --reload