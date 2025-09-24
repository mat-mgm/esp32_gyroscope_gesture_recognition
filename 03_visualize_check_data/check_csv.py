import pandas as pd
import numpy as np

df = pd.read_csv("../02_capture_data/data/up.csv")
print("UP")
print(df.isna().sum())   # count NaNs per column
print(df.describe())     # check min/max/mean/std

# Check for infinite values
print(np.isinf(df[['gx','gy','gz','ax','ay','az']].to_numpy()).sum())

df = pd.read_csv("../02_capture_data/data/down.csv")
print("DOWN")
print(df.isna().sum())   # count NaNs per column
print(df.describe())     # check min/max/mean/std

# Check for infinite values
print(np.isinf(df[['gx','gy','gz','ax','ay','az']].to_numpy()).sum())

df = pd.read_csv("../02_capture_data/data/right.csv")
print("RIGHT")
print(df.isna().sum())   # count NaNs per column
print(df.describe())     # check min/max/mean/std

# Check for infinite values
print(np.isinf(df[['gx','gy','gz','ax','ay','az']].to_numpy()).sum())

df = pd.read_csv("../02_capture_data/data/left.csv")
print("LEFT")
print(df.isna().sum())   # count NaNs per column
print(df.describe())     # check min/max/mean/std

# Check for infinite values
print(np.isinf(df[['gx','gy','gz','ax','ay','az']].to_numpy()).sum())

df = pd.read_csv("../02_capture_data/data/front.csv")
print("FRONT")
print(df.isna().sum())   # count NaNs per column
print(df.describe())     # check min/max/mean/std

# Check for infinite values
print(np.isinf(df[['gx','gy','gz','ax','ay','az']].to_numpy()).sum())
