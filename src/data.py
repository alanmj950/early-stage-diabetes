"""
src/data.py
Loads the raw CSV and writes a timestamped copy to data/processed
Usage:
    python src/data.py --input data/raw/diabetes_data_upload.csv --out data/processed/raw_loaded.csv
"""
import argparse
import pandas as pd
from datetime import datetime
import os

def load_raw(path: str) -> pd.DataFrame:
    """Load CSV and trim whitespace from column names."""
    df = pd.read_csv(path)
    # normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to raw csv")
    parser.add_argument("--out", required=True, help="path to save processed raw csv")
    args = parser.parse_args()

    df = load_raw(args.input)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[{datetime.now().isoformat()}] Loaded raw data shape: {df.shape} -> saved to {args.out}")
