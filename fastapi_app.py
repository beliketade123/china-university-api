from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="China University Classification API", version="1.0")

# Load artifacts
try:
    artifacts = joblib.load("china_university_model.joblib")
    model = artifacts['model']
    le_category = artifacts['le_category']
    le_province = artifacts['le_province']
except Exception as e:
    artifacts = None

class PredictionInput(BaseModel):
    national_rank: int = Field(..., example=10)
    score: float = Field(..., example=620.8)
    category: str = Field(..., example="comprehensive")
    province: str = Field(..., example="Beijing")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": artifacts is not None
    }

@app.post("/predict")
def predict_tier(data: PredictionInput):
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Model file not loaded.")
        
    if data.category not in le_category.classes_:
        raise HTTPException(status_code=400, detail=f"Category must be one of {list(le_category.classes_)}")
    if data.province not in le_province.classes_:
        raise HTTPException(status_code=400, detail=f"Province must be one of {list(le_province.classes_)}")
        
    cat_enc = le_category.transform([data.category])[0]
    prov_enc = le_province.transform([data.province])[0]
    
    input_df = pd.DataFrame([[data.national_rank, data.score, cat_enc, prov_enc]],
                            columns=['rank_clean', 'score_clean', 'category_encoded', 'province_encoded'])
    
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0].tolist()
    
    return {
        "predicted_tier": prediction,
        "probabilities": dict(zip(model.classes_, probs))
    }