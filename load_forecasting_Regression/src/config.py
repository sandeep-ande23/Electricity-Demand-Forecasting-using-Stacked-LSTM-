from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "figures"

LOOKBACK = 24
TEST_SIZE = 0.20
RANDOM_SEED = 42
EPOCHS = 50
BATCH_SIZE = 32

# Update these if your CSV uses different names.
COLUMN_ALIASES = {
    "datetime": ["datetime", "date", "timestamp", "time"],
    "load": ["load", "load_mw", "electricity_load", "demand", "demand_mw"],
    "temperature": ["temperature", "temp", "temperature_c"],
    "humidity": ["humidity", "humidity_percent", "relative_humidity"],
}
