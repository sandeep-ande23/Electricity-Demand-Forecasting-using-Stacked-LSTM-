import argparse
from pathlib import Path
import joblib
import numpy as np
import tensorflow as tf

from config import DATA_DIR, MODEL_DIR, LOOKBACK
from data_utils import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    csv_path = Path(args.data) if args.data else next(DATA_DIR.glob("*.csv"), None)
    if csv_path is None:
        raise FileNotFoundError("No CSV found in data/.")

    model = tf.keras.models.load_model(MODEL_DIR / "load_forecasting_lstm.keras")
    feature_scaler = joblib.load(MODEL_DIR / "feature_scaler.joblib")
    target_scaler = joblib.load(MODEL_DIR / "target_scaler.joblib")

    df = load_dataset(csv_path)

    if len(df) < LOOKBACK:
        raise ValueError(f"Need at least {LOOKBACK} observations.")

    latest = df[["load", "temperature", "humidity"]].tail(LOOKBACK).values
    latest_scaled = feature_scaler.transform(latest)
    X = latest_scaled.reshape(1, LOOKBACK, 3)

    prediction_scaled = model.predict(X, verbose=0)
    prediction = target_scaler.inverse_transform(prediction_scaled)[0, 0]

    last_time = df["datetime"].iloc[-1]
    print(f"Last observed timestamp : {last_time}")
    print(f"Predicted next load     : {prediction:.3f}")

if __name__ == "__main__":
    main()
