India Electricity Load Forecasting with LSTM
A time-series machine learning project that forecasts electricity load using historical load observations and weather-related features.

Project Overview
This project builds an LSTM-based forecasting pipeline for electricity demand. The model learns patterns from a rolling 24-hour historical window and predicts the next load value.

Core idea
Historical Load + Temperature + Humidity
                 ↓
        Data preprocessing
                 ↓
       24-hour input sequences
                 ↓
            LSTM model
                 ↓
       Next-load prediction
                 ↓
      MAE / RMSE / MAPE
                 ↓
      Actual vs Predicted
Why LSTM?
Electricity demand is a time-dependent problem. The load at one hour is related to previous hours, daily cycles, weather, and other temporal patterns.

An LSTM (Long Short-Term Memory) network is a recurrent neural network designed to learn dependencies in sequential data while reducing the vanishing-gradient problem associated with traditional RNNs.

Dataset
The original project uses an Indian electricity-load dataset with weather-related fields. The original notebook refers to the data file as:

Indian_Load_With_Synthetic_Weather.csv

Important: the weather features in the supplied project are described as synthetic weather data. This repository does not claim they are observations from a weather station.

Expected columns:

datetime / timestamp field
electricity load / demand in MW
temperature in °C
humidity in %
The training script automatically attempts to identify common names for these columns. If your CSV uses different names, update the column mapping in src/config.py.

Model Architecture
The baseline model follows the architecture in the supplied notebook:

Input: 24 consecutive hourly observations
Features per timestep: load, temperature, humidity
LSTM layer: 64 units
LSTM layer: 32 units
Dense output: 1 neuron
Loss: Mean Squared Error
Optimizer: Adam
Early stopping: enabled
The output is the predicted electricity load for the next timestep.

Evaluation
The project reports:

MAE (Mean Absolute Error): average absolute difference between actual and predicted load.
RMSE (Root Mean Squared Error): square root of average squared error; larger errors receive more penalty.
MAPE (Mean Absolute Percentage Error): average percentage error, expressed as a percentage.
Do not describe MAPE-derived values as "accuracy" without qualification. For a forecasting model, error metrics are the safer and more technically correct way to report performance.

Project Structure
load-forecasting-lstm/
│
├── data/
│   └── Indian_Load_With_Synthetic_Weather.csv
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── load_prediction_original.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── train.py
│   └── predict.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
Installation
git clone <your-repository-url>
cd load-forecasting-lstm

python -m venv .venv
Windows:

.venv\Scripts\activate
macOS/Linux:

source .venv/bin/activate
Install dependencies:

pip install -r requirements.txt
Run
Place the dataset inside data/, then run:

python src/train.py
The trained model and preprocessing artifacts are written to models/.

For a saved-model prediction:

python src/predict.py
Reproducibility
The pipeline:

Reads the time-series dataset.
Parses timestamps.
Sorts observations chronologically.
Selects load and weather features.
Handles missing values.
Scales model inputs.
Builds rolling 24-hour sequences.
Splits data chronologically.
Trains the LSTM.
Evaluates predictions using forecasting error metrics.
Saves the model and scalers.
Generates an actual-vs-predicted plot.
Important modeling decision: chronological split
Time-series data should not be randomly shuffled before the train/test split because future observations must not be allowed to influence training.

A chronological split better represents the real forecasting situation:

Past -----------------------> Future
Training                     Testing
Limitations
This is a strong portfolio-level forecasting project, but it should not be presented as a production utility forecasting system.

Current limitations include:

Weather variables are synthetic in the supplied dataset.
Only a limited set of features is used.
No holiday/calendar features are included.
No hyperparameter optimization is included.
The baseline predicts one future timestep.
Production deployment, monitoring, drift detection, and retraining are outside the current scope.
Future Improvements
Possible extensions:

Add real weather observations.
Add hour-of-day, day-of-week, month, weekend, and holiday features.
Compare LSTM against persistence, linear regression, Random Forest, XGBoost, and GRU baselines.
Tune sequence length.
Use walk-forward validation.
Predict 24 hours ahead rather than only one timestep.
Build a FastAPI inference service.
Containerize with Docker.
Deploy to AWS.
Add experiment tracking and model monitoring.
