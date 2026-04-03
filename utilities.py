import pandas as pd
import numpy as np

def clean_numeric(x):
    """Converts strings like 'A', '-', or blanks into NaN and ensures float type."""
    if pd.isna(x):
        return np.nan
    try:
        if isinstance(x, str):
            s = x.strip().lower()
            # Handle common 'null' strings
            if s in ("absent", "a", "nan", "none", "-", "", "nil"):
                return np.nan
            # Clean special characters often found in manual entries
            s = s.replace(",", "").replace("~", "").replace(">", "").replace("_", "").strip()
            return float(s)
        return float(x)
    except (ValueError, TypeError):
        return np.nan

def attendance_to_num(s):
    """Converts textual attendance info into a numeric scale (0.0–1.0)."""
    if pd.isna(s):
        return 0.5  # Default to 50% if unknown
    
    s = str(s).lower().strip()
    if any(word in s for word in ["almost", "full", "100", "90"]):
        return 1.0
    if "70-80" in s or "good" in s:
        return 0.8
    if "average" in s or "50" in s:
        return 0.5
    if any(word in s for word in ["very few", "low", "rare"]):
        return 0.2
    
    # Try to extract a number if it's like "75%"
    try:
        nums = [int(i) for i in s.split() if i.isdigit()]
        if nums:
            return nums[0] / 100 if nums[0] > 1 else nums[0]
    except:
        pass
        
    return 0.5

def resource_to_cat(x):
    """Simplifies resource types into 4 main categories."""
    if pd.isna(x):
        return "other"
    x = str(x).lower()
    if "youtube" in x or "video" in x:
        return "youtube"
    if "notes" in x or "ppt" in x or "slide" in x:
        return "notes"
    if "book" in x or "textbook" in x or "library" in x:
        return "book"
    return "other"

def scale_midsem_to_endsem(midsem, mid_out, total_out):
    """Scales midsem marks to estimated total. Added safety for 0 division."""
    try:
        val = clean_numeric(midsem)
        if pd.isna(val) or mid_out == 0:
            return np.nan
        # Proportional scaling: (Score / Max) * TotalPossible
        return (val / mid_out) * total_out
    except:
        return np.nan

# Grading thresholds (Percentile based)
RELATIVE_GRADE_BANDS = [
    ("O", 0.10),   # Outstanding (Top 10%)
    ("A+", 0.15),
    ("A", 0.15),
    ("B+", 0.20),
    ("B", 0.15),
    ("C", 0.15),
    ("P", 0.05),   # Pass
    ("F", 0.05)    # Fail
]

def assign_relative_grades(series):
    """Assigns grades based on percentile ranks (relative grading)."""
    # Filter out NaNs to calculate ranks properly
    valid_series = series.dropna()
    
    if len(valid_series) == 0:
        return pd.Series([], dtype="object")

    # rank(ascending=False) means highest score gets rank 1
    # pct=True scales ranks from 0.0 to 1.0
    ranks = valid_series.rank(method="first", pct=True, ascending=False)
    
    # Pre-calculate cumulative thresholds
    cum_thresholds = np.cumsum([p for _, p in RELATIVE_GRADE_BANDS])
    labels = [lbl for lbl, _ in RELATIVE_GRADE_BANDS]

    def map_pct(p):
        for i, thresh in enumerate(cum_thresholds):
            if p <= thresh:
                return labels[i]
        return labels[-1]

    return ranks.apply(map_pct)
