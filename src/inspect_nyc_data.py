import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Locate the NYC Yellow Taxi dataset
# --------------------------------------------------

DATA_PATH = Path("data/raw/yellow_tripdata_2025-01.parquet")

print("=" * 60)
print("NYC YELLOW TAXI DATA - INITIAL INSPECTION")
print("=" * 60)

# --------------------------------------------------
# 2. Check that the file exists
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

print("\nDataset found successfully!")
print(f"File: {DATA_PATH}")

# --------------------------------------------------
# 3. Load the Parquet dataset
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_parquet(DATA_PATH)

print("Dataset loaded successfully!")

# --------------------------------------------------
# 4. Dataset dimensions
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET SIZE")
print("=" * 60)

print(f"Number of rows: {len(df):,}")
print(f"Number of columns: {len(df.columns)}")

# --------------------------------------------------
# 5. Column names
# --------------------------------------------------

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# --------------------------------------------------
# 6. First five rows
# --------------------------------------------------

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head())

# --------------------------------------------------
# 7. Data types
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

# --------------------------------------------------
# 8. Missing values
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum().sort_values(ascending=False)

print(missing)

# --------------------------------------------------
# 9. Pickup date range
# --------------------------------------------------

if "tpep_pickup_datetime" in df.columns:

    print("\n" + "=" * 60)
    print("PICKUP DATE RANGE")
    print("=" * 60)

    print("Earliest pickup:", df["tpep_pickup_datetime"].min())
    print("Latest pickup:", df["tpep_pickup_datetime"].max())

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)