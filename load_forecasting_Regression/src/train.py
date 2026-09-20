import argparse
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

from config import DATA_DIR, MODEL_DIR, REPORT_DIR, LOOKBACK, EPOCHS, BATCH_SIZE, RANDOM_SEED
from data_utils import load_dataset, prepare_data

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def build_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 3)),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="CSV path")
    args = parser.parse_args()

    csv_path = Path(args.data) if args.data else next(DATA_DIR.glob("*.csv"), None)
    if csv_path is None:
        raise FileNotFoundError("No CSV found in data/. Provide --data PATH.")

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(csv_path)
    X_train, X_test, y_train, y_test, ts_train, ts_test, feature_scaler, target_scaler = prepare_data(df)

    model = build_model()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.10,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        shuffle=False,
        verbose=1
    )

    pred_scaled = model.predict(X_test, verbose=0)
    pred = target_scaler.inverse_transform(pred_scaled).ravel()
    actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    nonzero = actual != 0
    mape = np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100

    print("\nEvaluation")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAPE: {mape:.3f}%")

    model.save(MODEL_DIR / "load_forecasting_lstm.keras")
    joblib.dump(feature_scaler, MODEL_DIR / "feature_scaler.joblib")
    joblib.dump(target_scaler, MODEL_DIR / "target_scaler.joblib")

    # Save metrics
    (MODEL_DIR / "metrics.txt").write_text(
        f"MAE={mae:.6f}\nRMSE={rmse:.6f}\nMAPE={mape:.6f}%\n"
        f"LOOKBACK={LOOKBACK}\nTRAIN_ROWS={len(X_train)}\nTEST_ROWS={len(X_test)}\n",
        encoding="utf-8"
    )

    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("LSTM Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "training_loss.png", dpi=150)
    plt.close()

    # Actual vs predicted
    plt.figure(figsize=(12, 5))
    plt.plot(ts_test, actual, label="Actual")
    plt.plot(ts_test, pred, label="Predicted")
    plt.xlabel("Datetime")
    plt.ylabel("Electricity Load")
    plt.title("Actual vs Predicted Electricity Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
