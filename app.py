import streamlit as st
import pandas as pd
import joblib

# Page Layout Configuration
st.set_page_config(page_title="Chinese University Tier Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Chinese University Prestige Tier Predictor")
st.write("MLOps Lab 09 Deployment: Interactive Streamlit Frontend using Scikit-Learn")

# Load Model Artifacts
@st.cache_resource
def load_artifacts():
    return joblib.load("china_university_model.joblib")

try:
    artifacts = load_artifacts()
    model = artifacts['model']
    le_category = artifacts['le_category']
    le_province = artifacts['le_province']
    
    # Input Form Layout
    col1, col2 = st.columns(2)
    
    with col1:
        national_rank = st.number_input("National Rank", min_value=1, max_value=600, value=15, step=1)
        score = st.number_input("University Score", min_value=0.0, max_value=1200.0, value=592.3, step=0.1)
        
    with col2:
        category = st.selectbox("Category", options=list(le_category.classes_))
        province = st.selectbox("Province", options=list(le_province.classes_))
    
    if st.button("Predict University Prestige Tier"):
        cat_encoded = le_category.transform([category])[0]
        prov_encoded = le_province.transform([province])[0]
        
        input_data = pd.DataFrame([[national_rank, score, cat_encoded, prov_encoded]], 
                                  columns=['rank_clean', 'score_clean', 'category_encoded', 'province_encoded'])
        
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_
        
        st.subheader("Prediction Result")
        st.success(f"**Predicted Status:** {prediction}")
        
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({"Tier": classes, "Probability": probabilities})
        st.bar_chart(prob_df.set_index("Tier"))
        
except FileNotFoundError:
        st.error("Model file 'china_university_model.joblib' not found. Please run `python train_model.py` first.")