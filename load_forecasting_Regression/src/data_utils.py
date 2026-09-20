from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .config import COLUMN_ALIASES, LOOKBACK, TEST_SIZE

def find_column(df, aliases):
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in normalized:
            return normalized[alias.lower()]
    return None

def load_dataset(path):
    df = pd.read_csv(path)

    datetime_col = find_column(df, COLUMN_ALIASES["datetime"])
    load_col = find_column(df, COLUMN_ALIASES["load"])
    temp_col = find_column(df, COLUMN_ALIASES["temperature"])
    humidity_col = find_column(df, COLUMN_ALIASES["humidity"])

    missing = []
    for name, col in {
        "datetime": datetime_col,
        "load": load_col,
        "temperature": temp_col,
        "humidity": humidity_col,
    }.items():
        if col is None:
            missing.append(name)

    if missing:
        raise ValueError(
            f"Could not identify required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[datetime_col, load_col, temp_col, humidity_col]].copy()
    df.columns = ["datetime", "load", "temperature", "humidity"]

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in ["load", "temperature", "humidity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (
        df.dropna()
          .drop_duplicates(subset="datetime")
          .sort_values("datetime")
          .reset_index(drop=True)
    )

    return df

def make_sequences(values, target, lookback=LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i-lookback:i])
        y.append(target[i])
    return np.asarray(X), np.asarray(y)

def prepare_data(df):
    features = df[["load", "temperature", "humidity"]].values
    target = df["load"].values.reshape(-1, 1)

    split_index = int(len(df) * (1 - TEST_SIZE))

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Fit only on training portion to avoid leakage.
    feature_scaler.fit(features[:split_index])
    target_scaler.fit(target[:split_index])

    scaled_features = feature_scaler.transform(features)
    scaled_target = target_scaler.transform(target).ravel()

    X, y = make_sequences(scaled_features, scaled_target)

    # Sequence target timestamps correspond to y rows.
    timestamps = df["datetime"].iloc[LOOKBACK:].reset_index(drop=True)

    sequence_split = split_index - LOOKBACK
    X_train, X_test = X[:sequence_split], X[sequence_split:]
    y_train, y_test = y[:sequence_split], y[sequence_split:]
    ts_train, ts_test = timestamps[:sequence_split], timestamps[sequence_split:]

    return X_train, X_test, y_train, y_test, ts_train, ts_test, feature_scaler, target_scaler
