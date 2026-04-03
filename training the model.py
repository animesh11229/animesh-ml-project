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
    "Python": {"mid_col": "Python_20", "mid_out": 20, "total_out": 60},
    "Physics": {"mid_col": "Physics_20", "mid_out": 20, "total_out": 60},
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
        X =
