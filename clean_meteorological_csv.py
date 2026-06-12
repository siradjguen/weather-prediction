import pandas as pd


df = pd.read_csv("meteorological_data.csv")
print(" Loaded dataset")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Columns: {df.columns.tolist()}\n")


before = len(df)
df.drop_duplicates(inplace=True)
print(f"\n Duplicates removed: {before - len(df)}")


print(f"\n Missing values per column:")
print(df.isnull().sum().to_string())

df["Temperature"] = df["Temperature"].ffill()
df["Humidity"]    = df["Humidity"].ffill()
df["Wind Speed"]  = df["Wind Speed"].ffill()

print(f"\n   After forward fill — remaining nulls:")
print(df.isnull().sum().to_string())


df["Date"] = pd.to_datetime(df["Date"])
numeric_cols = ["Temperature", "Humidity", "Wind Speed", "Pressure", "Solar Radiation", "Rainfall"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
print(f"\n Step 3 — Data types confirmed:")
print(df.dtypes.to_string())


print(f"\n  Outlier detection:")

invalid_temp = df[(df["Temperature"] < -50) | (df["Temperature"] > 60)]
print(f"   Temperature outliers (outside -50 to 60°C): {len(invalid_temp)}")
if len(invalid_temp) > 0:
    print(f"   Values: {invalid_temp['Temperature'].values[:5]}")
df = df[(df["Temperature"] >= -50) & (df["Temperature"] <= 60)]

invalid_hum = df[(df["Humidity"] < 0) | (df["Humidity"] > 100)]
print(f"   Humidity outliers (outside 0–100%): {len(invalid_hum)}")

invalid_wind = df[df["Wind Speed"] < 0]
print(f"   Negative Wind Speed values: {len(invalid_wind)}")

invalid_pres = df[(df["Pressure"] < 870) | (df["Pressure"] > 1084)]
print(f"   Pressure outliers (outside 870–1084 hPa): {len(invalid_pres)}")

invalid_solar = df[df["Solar Radiation"] < 0]
print(f"   Negative Solar Radiation values: {len(invalid_solar)}")

invalid_rain = df[df["Rainfall"] < 0]
print(f"   Negative Rainfall values: {len(invalid_rain)}")

df = df.sort_values("Date").reset_index(drop=True)
date_diffs = df["Date"].diff().dropna()
gaps = date_diffs[date_diffs != pd.Timedelta(hours=1)]
print(f"\n Step 5 — Time series gaps (expected 1hr between readings):")
print(f"   Gaps found: {len(gaps)}")


for col in numeric_cols:
    df[col] = df[col].round(2)
print(f"\n  Numeric columns rounded to 2 decimal places")


df.reset_index(drop=True, inplace=True)
print(f"\n  Index reset")


df.to_csv("meteorological_data_cleaned.csv", index=False)
print(f"\n Done! Cleaned file saved as 'meteorological_data_cleaned.csv'")
print(f"   Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nPreview:")
print(df.head(10).to_string(index=False))

print(f"\n Summary statistics:")
print(df[numeric_cols].describe().round(2).to_string())
