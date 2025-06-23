from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI(title="Obesity Level Predictor")

class ObesityFeatures(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

@app.post("/predict")
def predict_obesity(features: ObesityFeatures):
    data_dict = features.dict()

    categorical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC",
                           "SMOKE", "SCC", "CALC", "MTRANS"]
    for col in categorical_columns:
        data_dict[col] = str(data_dict[col]).strip()

    input_df = pd.DataFrame([data_dict])
    print(input_df.dtypes)
    print(input_df)

    try:
        prediction_encoded = model.predict(input_df)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        return {
            "prediction_encoded": int(prediction_encoded),
            "prediction_label": prediction_label
        }

    except Exception as e:
        return {
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fast_api:app", host="0.0.0.0", port=8000)