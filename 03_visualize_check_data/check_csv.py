import pandas as pd
import numpy as np
import os

DATA_DIR = "../02_capture_data/data"

def analyze_movement(label, filename):
    """Load CSV, print NaN counts, summary stats, and infinite values."""
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)

    print(f"\n=== {label.upper()} ===")
    print("NaN counts per column:")
    print(df.isna().sum())
    print("\nSummary statistics:")
    print(df.describe())

    # Check for infinite values in sensor columns
    inf_count = np.isinf(df[['gx','gy','gz','ax','ay','az']].to_numpy()).sum()
    print(f"\nInfinite values in sensor columns: {inf_count}")

    return df

# Movements to analyze
movements = {
    "UP": "up.csv",
    "DOWN": "down.csv",
    "RIGHT": "right.csv",
    "LEFT": "left.csv",
    "FRONT": "front.csv",
    "STILL": "still.csv"   # new movement
}

# Run analysis for each movement
for label, filename in movements.items():
    analyze_movement(label, filename)

