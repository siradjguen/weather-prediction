import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
FILES = {
    "Forward Fill":    "meteorological_data_cleaned.csv",
    "Interpolation":   "meteorological_data_cleaned_INTERPOLATION.csv",
    "KNN":             "meteorological_data_cleaned_KNN.csv"
}

SEQUENCE_LENGTH = 24      # use last 24 hours to predict next hour
TARGET_COL      = "Temperature"
NUMERIC_COLS    = ["Temperature", "Humidity", "Wind Speed",
                   "Pressure", "Solar Radiation", "Rainfall"]
TEST_SIZE       = 0.2     # last 20% of data for testing
EPOCHS          = 50
BATCH_SIZE      = 32

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def create_sequences(data, target, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(target[i + seq_len])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def evaluate_model(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2

# ─────────────────────────────────────────
# MAIN LOOP — Train on each file
# ─────────────────────────────────────────
results = {}
histories = {}

for technique, filepath in FILES.items():
    print(f"\n{'='*50}")
    print(f"  Training on: {technique}")
    print(f"{'='*50}")

    # Load
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Scale
    scaler      = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[NUMERIC_COLS])

    # Scale target separately to inverse transform later
    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(
        df[[TARGET_COL]]
    ).flatten()

    # Create sequences
    X, y = create_sequences(scaled_data, target_scaled, SEQUENCE_LENGTH)

    # Split — keep time order, no shuffling
    split      = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples:  {len(X_test)}")

    # Build and train
    model = build_model((SEQUENCE_LENGTH, len(NUMERIC_COLS)))
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    histories[technique] = history.history

    # Predict and inverse transform
    y_pred_scaled = model.predict(X_test)
    y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    # Evaluate
    mae, rmse, r2 = evaluate_model(y_true, y_pred)
    results[technique] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    print(f"\n   Results for {technique}:")
    print(f"   MAE:  {mae:.4f}°C")
    print(f"   RMSE: {rmse:.4f}°C")
    print(f"   R²:   {r2:.4f}")

# ─────────────────────────────────────────
# FINAL COMPARISON TABLE
# ─────────────────────────────────────────
print(f"\n{'='*50}")
print("  FINAL COMPARISON")
print(f"{'='*50}")
print(f"{'Technique':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 50)
for technique, metrics in results.items():
    print(f"{technique:<20} {metrics['MAE']:>8.4f} "
          f"{metrics['RMSE']:>8.4f} {metrics['R2']:>8.4f}")

best = min(results, key=lambda x: results[x]["RMSE"])
print(f"\n Best technique: {best} "
      f"(RMSE = {results[best]['RMSE']:.4f}°C)")

# ─────────────────────────────────────────
# PLOT — Training loss comparison
# ─────────────────────────────────────────
plt.figure(figsize=(12, 5))
for technique, history in histories.items():
    plt.plot(history["val_loss"], label=technique)
plt.title("Validation Loss per Technique")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.tight_layout()
plt.savefig("comparison_loss.png")
print("\n Loss chart saved as 'comparison_loss.png'")
