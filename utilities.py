

import pandas as pd
import numpy as np

def clean_numeric(x):
    """Converts strings like 'A', '-', or blanks into NaN and ensures float type."""
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("absent", "a", "nan", "none", "-", ""):
                return np.nan
            s = s.replace(",", "").replace("~", "").replace(">", "").replace("_", "").strip()
            return float(s)
        return float(x)
    except:
        return np.nan

def attendance_to_num(s):
    """Converts textual attendance info into a numeric scale (0.0–1.0)."""
    if pd.isna(s):
        return np.nan
    s = str(s).lower()
    if "almost" in s:
        return 1.0
    if "70-80" in s:
        return 0.8
    if "very few" in s:
        return 0.3
    return 0.5

def resource_to_cat(x):
    """Simplifies resource types (YouTube, Notes, Book, Other)."""
    if pd.isna(x):
        return "other"
    x = str(x).lower()
    if "youtube" in x:
        return "youtube"
    if "notes" in x:
        return "notes"
    if "book" in x:
        return "book"
    return "other"


def scale_midsem_to_endsem(midsem, mid_out, total_out):
    """Scales midsem marks to estimated total using proportional ratio."""
    try:
        if pd.isna(midsem):
            return np.nan
        return (midsem / mid_out) * total_out
    except:
        return np.nan

RELATIVE_GRADE_BANDS = [
    ("A+", 0.10),
    ("A", 0.15),
    ("A-", 0.15),
    ("B+", 0.20),
    ("B", 0.15),
    ("C", 0.15),
    ("D", 0.05),
    ("F", 0.05)
]

def assign_relative_grades(series):
    """Assigns grades based on percentile ranks (relative grading)."""
    series = series.dropna()
    if len(series) == 0:
        return pd.Series([], dtype="float64")

    ranks = series.rank(method="first", pct=True, ascending=False)
    cum = np.cumsum([p for _, p in RELATIVE_GRADE_BANDS])
    labels = [lbl for lbl, _ in RELATIVE_GRADE_BANDS]

    def map_pct(p):
        for i, thresh in enumerate(cum):
            if p <= thresh:
                return labels[i]
        return labels[-1]

    return ranks.apply(map_pct)
