
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from utils import (
    clean_numeric,
    attendance_to_num,
    resource_to_cat,
    assign_relative_grades,
    scale_midsem_to_endsem
)

CSV_PATH = "cleaned_dataset.csv"  
SUBJECT_CONFIG = {
    "Math": {"mid_col": "Math_25", "mid_out": 25, "end_out": 50, "total_out": 75},
    "Python": {"mid_col": "Python_25", "mid_out": 25, "end_out": 50, "total_out": 75},
    "Physics": {"mid_col": "Physics_20", "mid_out": 20, "end_out": 40, "total_out": 60},
    "EG": {"mid_col": "EG_20", "mid_out": 20, "end_out": 40, "total_out": 60}
}
RANDOM_STATE = 42
# -----------------------------------------------------------

print("🔍 Loading dataset...")
df = pd.read_csv(CSV_PATH)

# Clean numeric columns (convert text or blanks into numbers/NaN)
for cfg in SUBJECT_CONFIG.values():
    col = cfg["mid_col"]
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# Convert Attendance & Resource columns into numeric/categorical codes
if "Attendance" in df.columns:
    df["Attendance_num"] = df["Attendance"].apply(attendance_to_num)
else:
    df["Attendance_num"] = np.nan

if "Resource" in df.columns:
    df["Resource_cat"] = df["Resource"].apply(resource_to_cat)
else:
    df["Resource_cat"] = "other"

# Fill any missing faculty ratings with 0
for rcol in ["ECE_Rate", "PHY_Rate", "EG_Rate", "PY_Rate", "MATH_Rate"]:
    if rcol in df.columns:
        df[rcol] = df[rcol].fillna(0)
    else:
        df[rcol] = 0

models_info = {}


for subj, cfg in SUBJECT_CONFIG.items():
    mid_col = cfg["mid_col"]
    if mid_col not in df.columns:
        print(f"⚠️ Skipping {subj}: column {mid_col} not present in dataset.")
        continue

    # (1) Scale midsem marks to approximate full total marks
    df[f"{mid_col}_est_total"] = df[mid_col].apply(
        lambda x: scale_midsem_to_endsem(x, cfg["mid_out"], cfg["total_out"])
    )

    # (2) Assign relative grades using percentiles
    try:
        grade_series = assign_relative_grades(df[f"{mid_col}_est_total"].dropna())
        df.loc[grade_series.index, f"{subj}_Grade"] = grade_series.values
    except Exception as e:
        print(f"⚠️ Could not assign grades for {subj}: {e}")
        df[f"{subj}_Grade"] = np.nan

    # (3) Prepare training data
    train_df = df[~df[f"{subj}_Grade"].isna() & ~df[mid_col].isna()].copy()
    if len(train_df) < 3:
        print(f"⚠️ Not enough training rows for {subj} ({len(train_df)}). Creating model anyway.")
    
    # (4) Create features
    features = []
    train_df[f"{mid_col}_pctile"] = train_df[mid_col].rank(pct=True)
    train_df[f"{mid_col}_zscore"] = (train_df[mid_col] - train_df[mid_col].mean()) / (
        train_df[mid_col].std(ddof=0) + 1e-9
    )
    features += [
        mid_col,
        f"{mid_col}_pctile",
        f"{mid_col}_zscore",
        "Attendance_num",
        "PY_Rate",
        "PHY_Rate",
        "EG_Rate",
        "MATH_Rate"
    ]

    # (5) Add other subjects' midsem data
    for other_cfg in SUBJECT_CONFIG.values():
        other_col = other_cfg["mid_col"]
        if other_col != mid_col and other_col in df.columns:
            train_df[f"{other_col}_pctile"] = train_df[other_col].rank(pct=True)
            features += [other_col, f"{other_col}_pctile"]

    # (6) One-hot encode Resource column
    X = train_df[features].copy()
    X = pd.concat([X, pd.get_dummies(train_df["Resource_cat"], prefix="Res", dummy_na=True)], axis=1)
    X = X.fillna(X.median(numeric_only=True))
    y = train_df[f"{subj}_Grade"].astype(str)

    # (7) Split into train/test sets for accuracy checking
    if len(train_df) > 4:  # ensure enough data to split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    # (8) Train the model
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # (9) Test predictions
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n✅ Model trained for {subj}")
    print(f"📊 Accuracy: {acc * 100:.2f}% ({len(y_test)} test samples)")
    print("📄 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("---------------------------------------------------")

    # (10) Save model
    model_obj = {
        "model": clf,
        "features": X.columns.tolist(),
        "mid_col": mid_col,
        "mid_out": cfg["mid_out"],
        "total_out": cfg["total_out"]
    }
    fname = f"{subj}_grade_model.pkl"
    joblib.dump(model_obj, fname)
    models_info[subj] = {
        "rows": len(train_df),
        "accuracy": acc,
        "features": X.columns.tolist()
    }

print("\n🎓 Training complete. Models saved in working directory.")
print("Summary of results:")
for subj, info in models_info.items():
    print(f" - {subj}: {info['rows']} rows, accuracy = {info['accuracy']*100:.2f}%")
