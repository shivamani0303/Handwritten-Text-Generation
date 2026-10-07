import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

csv_path = BASE_DIR / "dataset" / "english.csv"

df = pd.read_csv(csv_path)

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nTotal records:")
print(len(df))

print("\nUnique labels:")
print(df["label"].nunique())

print("\nSample labels:")
print(df["label"].unique()[:20])