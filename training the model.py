import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utilities import (
    clean_numeric,
    attendance_to_num,
    resource_to_cat,
    assign_relative_grades,
    scale_midsem_to_endsem
)

CSV_PATH = "cleaned_dataset.csv"  

# Revised to match project dataset.csv columns
SUBJECT_CONFIG = {
    "Math": {"mid_col": "Math_25", "mid_out": 25, "total_out": 75},
    "ECE": {"mid_col": "ECE_20", "mid_out": 20, "total_out": 60},
    "Clanguage": {"mid_col": "Clanguage_20", "mid_out": 20, "total_out": 60},
    "EG": {"mid_col": "EG_20", "mid_out": 20, "total_out": 60}
}
RANDOM_STATE = 42

def train():
    print("🔍 Loading dataset...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"❌ Error: {CSV_PATH} not found. Please create it first.")
        return

    # Pre-processing
    for subj, cfg in SUBJECT_CONFIG.items():
        col = cfg["mid_col"]
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    df["Attendance_num"] = df["Attendance"].apply(attendance_to_num) if "Attendance" in df.columns else 0.5
    df["Resource_cat"] = df["Resource"].apply(resource_to_cat) if "Resource" in df.columns else "other"

    models_info = {}

    for subj, cfg in SUBJECT_CONFIG.items():
        mid_col = cfg["mid_col"]
        if mid_col not in df.columns:
            continue

        # Scale and Grade
        df[f"{mid_col}_est_total"] = df[mid_col].apply(lambda x: scale_midsem_to_endsem(x, cfg["mid_out"], cfg["total_out"]))
        
        valid_indices = df[f"{mid_col}_est_total"].dropna().index
        if len(valid_indices) < 5:
            continue

        grade_series = assign_relative_grades(df.loc[valid_indices, f"{mid_col}_est_total"])
        df.loc[grade_series.index, f"{subj}_Grade"] = grade_series.values

        # Filter data for training
        train_df = df[df[f"{subj}_Grade"].notna() & df[mid_col].notna()].copy()
        
        # Feature Engineering
        train_df[f"{mid_col}_pctile"] = train_df[mid_col].rank(pct=True)
        train_df[f"{mid_col}_zscore"] = (train_df[mid_col] - train_df[mid_col].mean()) / (train_df[mid_col].std(ddof=0) + 1e-9)
        
        base_features = [mid_col, f"{mid_col}_pctile", f"{mid_col}_zscore", "Attendance_num"]
        X = train_df[base_features].copy()
        
        # One-hot encode Resources
        resource_dummies = pd.get_dummies(train_df["Resource_cat"], prefix="Res")
        X = pd.concat([X, resource_dummies], axis=1)
        X = X.fillna(X.median(numeric_only=True))
        y = train_df[f"{subj}_Grade"].astype(str)

        # Reduced test_size for very small datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=RANDOM_STATE)

        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        
        # Save Model
        joblib.dump({
            "model": clf, 
            "features": X.columns.tolist(), 
            "mid_col": mid_col
        }, f"{subj}_grade_model.pkl")
        
        models_info[subj] = {"rows": len(train_df), "accuracy": accuracy_score(y_test, clf.predict(X_test))}

    print("\n🎓 Training complete.")
    for subj, info in models_info.items():
        print(f" - {subj}: {info['rows']} rows trained.")

if __name__ == "__main__":
    train()
