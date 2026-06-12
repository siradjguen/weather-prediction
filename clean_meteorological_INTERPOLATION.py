import pandas as pd
import numpy as np


df = pd.read_csv("meteorological_data.csv")
print(" Loaded dataset")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")


before = len(df)
df.drop_duplicates(inplace=True)
print(f" Step 1 — Duplicates removed: {before - len(df)}")


print(f"\nStep 2 — Missing values before interpolation:")
print(df.isnull().sum().to_string())


numeric_cols = ["Temperature", "Humidity", "Wind Speed", "Pressure", "Solar Radiation", "Rainfall"]
for col in numeric_cols:
    df[col] = df[col].interpolate(method="linear", limit_direction="forward")

print(f"\n   After interpolation — remaining nulls:")
print(df[numeric_cols].isnull().sum().to_string())


df["Date"] = pd.to_datetime(df["Date"])
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print(f"\n Step 3 — Data types confirmed:")
print(df.dtypes.to_string())


print(f"\n Step 4 — Outlier detection:")
invalid_temp = df[(df["Temperature"] < -50) | (df["Temperature"] > 60)]
print(f"   Temperature outliers: {len(invalid_temp)}")
df = df[(df["Temperature"] >= -50) & (df["Temperature"] <= 60)]

invalid_hum  = df[(df["Humidity"] < 0) | (df["Humidity"] > 100)]
print(f"   Humidity outliers: {len(invalid_hum)}")

invalid_wind = df[df["Wind Speed"] < 0]
print(f"   Negative Wind Speed: {len(invalid_wind)}")

invalid_pres = df[(df["Pressure"] < 870) | (df["Pressure"] > 1084)]
print(f"   Pressure outliers: {len(invalid_pres)}")

invalid_solar = df[df["Solar Radiation"] < 0]
print(f"   Negative Solar Radiation: {len(invalid_solar)}")

invalid_rain = df[df["Rainfall"] < 0]
print(f"   Negative Rainfall: {len(invalid_rain)}")


df = df.sort_values("Date").reset_index(drop=True)
date_diffs = df["Date"].diff().dropna()
gaps = date_diffs[date_diffs != pd.Timedelta(hours=1)]
print(f"\n Step 5 — Time gaps found: {len(gaps)}")


for col in numeric_cols:
    df[col] = df[col].round(2)
print(f"\n Step 6 — Rounded to 2 decimal places")


df.reset_index(drop=True, inplace=True)
print(f" Step 7 — Index reset")


df.to_csv("meteorological_data_cleaned_INTERPOLATION.csv", index=False)
print(f"\n Done! Saved as 'meteorological_data_cleaned_INTERPOLATION.csv'")
print(f"   Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nPreview:")
print(df.head(10).to_string(index=False))
print(f"\n Summary statistics:")
print(df[numeric_cols].describe().round(2).to_string())
