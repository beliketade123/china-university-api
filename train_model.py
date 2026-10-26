import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_save_model():
    # 1. Load Dataset
    df = pd.read_csv("Best-Global-Universities-China new.csv")
    
    # 2. Target Definition (Prestige Tier)
    def assign_tier(tag):
        t = str(tag)
        if '985' in t:
            return 'Project 985 (Top Elite)'
        elif '211' in t:
            return 'Project 211 (Key National)'
        else:
            return 'General Public University'

    df['prestige_tier'] = df['tags'].apply(assign_tier)
    
    # 3. Clean Features
    df['score_clean'] = pd.to_numeric(df['score'], errors='coerce')
    df['rank_clean'] = pd.to_numeric(df['national_rank'], errors='coerce')
    
    # Impute missing numeric values
    df['score_clean'] = df['score_clean'].fillna(df['score_clean'].median())
    df['rank_clean'] = df['rank_clean'].fillna(df['rank_clean'].max() + 10)
    
    # Filter valid categories
    valid_categories = ['stem', 'comprehensive', 'normal', 'agriculture', 'forestry']
    df['category_clean'] = df['category'].apply(lambda x: x if x in valid_categories else 'comprehensive')
    
    # Categorical Encoders
    le_category = LabelEncoder()
    df['category_encoded'] = le_category.fit_transform(df['category_clean'])
    
    le_province = LabelEncoder()
    df['province_encoded'] = le_province.fit_transform(df['province'].fillna('Unknown'))
    
    # Feature matrix & Target vector
    feature_cols = ['rank_clean', 'score_clean', 'category_encoded', 'province_encoded']
    X = df[feature_cols]
    y = df['prestige_tier']
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    
    # 6. Model Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Training Complete! Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # 7. Export Model and Artifacts
    artifacts = {
        'model': clf,
        'le_category': le_category,
        'le_province': le_province,
        'feature_names': feature_cols
    }
    joblib.dump(artifacts, "china_university_model.joblib")
    print("Saved pipeline artifacts to 'china_university_model.joblib'")

if __name__ == "__main__":
    train_and_save_model()