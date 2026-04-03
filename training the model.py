import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import (
    clean_numeric,
    attendance_to_num,
    resource_to_cat,
    assign_relative_grades,
    scale_midsem_to_endsem
)

CSV_PATH = "cleaned_dataset.csv"  
# Updated SUBJECT_CONFIG to match your available subjects
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
        print(f"❌ Error: {CSV_PATH} not found.")
        return

    # --- Pre-processing ---
    for subj, cfg in SUBJECT_CONFIG.items():
        col = cfg["mid_col"]
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    if "Attendance" in df.columns:
        df["Attendance_num"] = df["Attendance"].apply(attendance_to_num)
    else:
        df["Attendance_num"] = 0

    df["Resource_cat"] = df["Resource"].apply(resource_to_cat) if "Resource" in df.columns else "other"

    models_info = {}

    for subj, cfg in SUBJECT_CONFIG.items():
        mid_col = cfg["mid_col"]
        if mid_col not in df.columns:
            print(f"⚠️ Skipping {subj}: Column {mid_col} not found.")
            continue

        # (1) Scale and Grade
        df[f"{mid_col}_est_total"] = df[mid_col].apply(lambda x: scale_midsem_to_endsem(x, cfg["mid_out"], cfg["total_out"]))
        
        # Calculate grades (dropping NaNs for the calculation only)
        valid_indices = df[f"{mid_col}_est_total"].dropna().index
        if len(valid_indices) < 5:
            print(f"⚠️ Skipping {subj}: Not enough data points.")
            continue

        try:
            grade_series = assign_relative_grades(df.loc[valid_indices, f"{mid_col}_est_total"])
            df.loc[grade_series.index, f"{subj}_Grade"] = grade_series.values
        except Exception as e:
            print(f"❌ Grading error in {subj}: {e}")
            continue

        # (2) Filter data for training
        train_df = df[df[f"{subj}_Grade"].notna() & df[mid_col].notna()].copy()
        
        # (3) Feature Engineering
        train_df[f"{mid_col}_pctile"] = train_df[mid_col].rank(pct=True)
        train_df[f"{mid_col}_zscore"] = (train_df[mid_col] - train_df[mid_col].mean()) / (train_df[mid_col].std(ddof=0) + 1e-9)
        
        base_features = [mid_col, f"{mid_col}_pctile", f"{mid_col}_zscore", "Attendance_num"]

        # (4) One-hot encode Resources
        X = train_df[base_features].copy()
        resource_dummies = pd.get_dummies(train_df["Resource_cat"], prefix="Res")
        X = pd.concat([X, resource_dummies], axis=1)
        
        # Ensure all columns are numeric for the median fill
        X = X.fillna(X.median(numeric_only=True))
        y = train_df[f"{subj}_Grade"].astype(str)

        # (5) Train/Test Split (Added safety for stratify)
        class_counts = y.value_counts()
        can_stratify = all(class_counts > 1) and (len(train_df) > 10)
        
        split_data = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, 
            stratify=y if can_stratify else None
        )
        X_train, X_test, y_train, y_test = split_data

        # (6) Model Execution
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # (7) Save Model & the EXACT feature list
        # This is critical so the testing script knows which columns to build
        joblib.dump({
            "model": clf, 
            "features": X.columns.tolist(), 
            "mid_col": mid_col
        }, f"{subj}_grade_model.pkl")
        
        models_info[subj] = {"rows": len(train_df), "accuracy": acc}

    print("\n🎓 Training complete. Summary:")
    print("-" * 30)
    for subj, info in models_info.items():
        print(f" - {subj:8}: {info['rows']} rows, Accuracy: {info['accuracy']*100:.2f}%")

if __name__ == "__main__":
    train()
