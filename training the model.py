import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from utils import (
    clean_numeric,
    attendance_to_num,
    resource_to_cat,
    assign_relative_grades,
    scale_midsem_to_endsem
)

CSV_PATH = "cleaned_dataset.csv"  
SUBJECT_CONFIG = {
    "Math": {"mid_col": "Math_25", "mid_out": 25, "total_out": 75},
    "ECE": {"mid_col": "ECE_20", "mid_out": 20, "total_out": 65},
    "CLanguage": {"mid_col": "Clanguage_20", "mid_out": 20, "total_out": 60},
    "EG": {"mid_col": "EG_20", "mid_out": 20, "total_out": 60}
}
RANDOM_STATE = 42

print("🔍 Loading dataset...")
df = pd.read_csv(CSV_PATH)

# Basic Data Cleaning
for cfg in SUBJECT_CONFIG.values():
    col = cfg["mid_col"]
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

df["Attendance_num"] = df["Attendance"].apply(attendance_to_num) if "Attendance" in df.columns else np.nan
df["Resource_cat"] = df["Resource"].apply(resource_to_cat) if "Resource" in df.columns else "other"

models_info = {}

for subj, cfg in SUBJECT_CONFIG.items():
    mid_col = cfg["mid_col"]
    if mid_col not in df.columns:
        continue

    # (1) Scale and Grade
    df[f"{mid_col}_est_total"] = df[mid_col].apply(lambda x: scale_midsem_to_endsem(x, cfg["mid_out"], cfg["total_out"]))
    
    try:
        grade_series = assign_relative_grades(df[f"{mid_col}_est_total"].dropna())
        df.loc[grade_series.index, f"{subj}_Grade"] = grade_series.values
    except:
        df[f"{subj}_Grade"] = np.nan

    # (2) Filter data
    train_df = df[df[f"{subj}_Grade"].notna() & df[mid_col].notna()].copy()
    
    # (3) Feature Engineering
    train_df[f"{mid_col}_pctile"] = train_df[mid_col].rank(pct=True)
    train_df[f"{mid_col}_zscore"] = (train_df[mid_col] - train_df[mid_col].mean()) / (train_df[mid_col].std(ddof=0) + 1e-9)
    
    features = [mid_col, f"{mid_col}_pctile", f"{mid_col}_zscore", "Attendance_num"]

    # (4) One-hot encode Resources & handle NaNs
    X = train_df[features].copy()
    X = pd.concat([X, pd.get_dummies(train_df["Resource_cat"], prefix="Res")], axis=1)
    X = X.fillna(X.median(numeric_only=True))
    y = train_df[f"{subj}_Grade"].astype(str)

    # (5) Train/Test Split
    split_data = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y) if len(train_df) > 4 else (X, X, y, y)
    X_train, X_test, y_train, y_test = split_data

    # (6) Model Execution
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    # (7) Save Model
    joblib.dump({"model": clf, "features": X.columns.tolist(), "mid_col": mid_col}, f"{subj}_grade_model.pkl")
    models_info[subj] = {"rows": len(train_df), "accuracy": acc}

print("\n🎓 Training complete. Summary:")
for subj, info in models_info.items():
    print(f" - {subj}: {info['rows']} rows, accuracy = {info['accuracy']*100:.2f}%")
