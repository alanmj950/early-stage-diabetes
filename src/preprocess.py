"""
src/preprocess.py
Preprocess rules:
 - Convert many 1/2 -> 1/0 encodings (UCI dataset uses 1/2 for yes/no)
 - Convert Gender to binary (gender_M)
 - Encode target 'class' to 1/0
 - Impute Age (median) and other columns (mode) if missing
 - Save processed CSV
Usage:
    python src/preprocess.py --input data/processed/raw_loaded.csv --out data/processed/processed.csv
"""
import argparse
import pandas as pd
import numpy as np
import os

def map_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Map columns containing only {1,2} -> {1,0} and Yes/No strings -> {1,0}"""
    df = df.copy()
    exclude = set(['Age','Gender','class','age','gender'])
    for c in df.columns:
        if c in exclude:
            continue
        # skip non-scalar or object columns handled below
        vals = set(df[c].dropna().unique())
        # numeric encoding 1/2
        if vals.issubset({1,2}):
            df[c] = df[c].map({1:1, 2:0})
        # yes/no strings
        elif vals.issubset({'Yes','No','yes','no','YES','NO'}):
            df[c] = df[c].astype(str).str.lower().map({'yes':1,'no':0})
    return df

def handle_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Create gender_M (Male=1) and drop original Gender column"""
    df = df.copy()
    if 'Gender' in df.columns:
        vals = set(df['Gender'].dropna().unique())
        if vals.issubset({1,2}):
            # many UCI files use 1=Male, 2=Female
            df['gender_M'] = df['Gender'].map({1:1,2:0})
        else:
            df['gender_M'] = df['Gender'].astype(str).str.lower().map(lambda s: 1 if 'male' in s else 0)
        df = df.drop(columns=['Gender'])
    return df

def encode_label(df: pd.DataFrame) -> pd.DataFrame:
    """Encode 'class' to 1 positive, 0 negative"""
    df = df.copy()
    if 'class' in df.columns:
        vals = set(df['class'].dropna().unique())
        if vals.issubset({1,2}):
            df['class'] = df['class'].map({1:1,2:0})
        else:
            df['class'] = df['class'].astype(str).str.lower().map(lambda s: 1 if 'positive' in s or 'yes' in s else 0)
    else:
        raise KeyError("No 'class' column found in data")
    return df

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute Age with median, other columns with mode. Drop any remaining NA rows."""
    df = df.copy()
    # Age
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        median_age = int(df['Age'].median(skipna=True))
        df['Age'].fillna(median_age, inplace=True)
    # Others
    for c in df.columns:
        if c in ['Age','class']:
            continue
        if df[c].isna().sum() > 0:
            # mode might be empty if column all NaN
            mode_vals = df[c].mode(dropna=True)
            if len(mode_vals) > 0:
                df[c].fillna(mode_vals[0], inplace=True)
            else:
                df[c].fillna(0, inplace=True)
    # final drop any rows still with NA
    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = map_binary(df)
    df = handle_gender(df)
    df = encode_label(df)
    df = impute_missing(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Preprocessing complete. Saved processed data to {args.out}. Shape: {df.shape}")
