from flask import Flask, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import datetime, random
from scipy.fft import rfft, rfftfreq
from pymodbus.client.sync import ModbusTcpClient

app = Flask(__name__)

# ======= Load trained model + top features =======
model = joblib.load("pump_model_1h.joblib")
top_features = joblib.load("selected_features_1h.joblib")
le = joblib.load("label_encoder_1h.joblib")

# ======= Pumps & Sensor Columns =======
pumps = ["Gunhill", "Muyenga", "Katosi"]
sensor_cols = ["DE_motor", "NDE_pump", "Coupling"]

# ======= Helper: Generate new sensor snapshot =======
client = ModbusTcpClient("192.168.0.10")  # Replace with PLC IP

def get_sensor_data(pump_name):
    # Read registers (adjust register addresses according to your PLC mapping)
    flow_reg = client.read_holding_registers(0, 1)
    pressure_reg = client.read_holding_registers(1, 1)
    level_reg = client.read_holding_registers(2, 1)
    nde_motor_reg = client.read_holding_registers(3, 1)
    de_motor_reg = client.read_holding_registers(4, 1)
    de_pump_reg = client.read_holding_registers(5, 1)
    nde_pump_reg = client.read_holding_registers(6, 1)
    coupling_reg = client.read_holding_registers(7, 1)

    # Handle errors gracefully
    if flow_reg.isError():
        return None

    # Convert raw register values to engineering units (adjust scaling factors)
    flow = flow_reg.registers[0] / 10.0
    pressure = pressure_reg.registers[0] / 100.0
    level = level_reg.registers[0] / 100.0
    nde_motor = nde_motor_reg.registers[0] / 1000.0
    de_motor = de_motor_reg.registers[0] / 1000.0
    de_pump = de_pump_reg.registers[0] / 1000.0
    nde_pump = nde_pump_reg.registers[0] / 1000.0
    coupling = coupling_reg.registers[0] / 1000.0

    return {
        "Time": datetime.datetime.utcnow().isoformat(),
        "pump": pump_name,
        "flow": flow,
        "pressure": pressure,
        "level": level,
        "NDE_motor": nde_motor,
        "DE_motor": de_motor,
        "DE_pump": de_pump,
        "NDE_pump": nde_pump,
        "Coupling": coupling
    }


# ======= Helper: Preprocess new sensor data =======
def preprocess_new_data(new_df, top_features):
    """
    Preprocess incoming sensor data for inference.
    Expects columns: Time, flow, pressure, level, NDE_motor, DE_motor, DE_pump, NDE_pump, Coupling, pump
    """
    # --- Ensure datetime ---
    new_df["Time"] = pd.to_datetime(new_df["Time"], errors="coerce")
    new_df = new_df.set_index("Time")

    # --- Resample to 1min ---
    numeric_cols = ['flow', 'pressure', 'level', 'NDE_motor', 'DE_motor', 'DE_pump', 'NDE_pump', 'Coupling']
    resampled = new_df.resample('1min').agg(
        {col: ['mean', 'std', 'min', 'max'] for col in numeric_cols} |
        {'pump': 'last'}
    )

    resampled.columns = [
        '_'.join(col).strip() if isinstance(col, tuple) else col
        for col in resampled.columns
    ]

    # --- Outlier clipping ---
    def remove_outliers_iqr(series, factor=1.5):
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - factor*IQR, Q3 + factor*IQR
        return series.clip(lower, upper)

    for col in resampled.columns:
        if any(sensor in col for sensor in numeric_cols):
            resampled[col] = remove_outliers_iqr(resampled[col])

    # --- Rolling stats ---
    windows = [3, 5, 10]
    for col in [c for c in resampled.columns if any(sensor in c for sensor in numeric_cols)]:
        for w in windows:
            resampled[f"{col}_roll_mean_{w}"] = resampled[col].rolling(w, min_periods=1).mean()
            resampled[f"{col}_roll_std_{w}"]  = resampled[col].rolling(w, min_periods=1).std()
            resampled[f"{col}_roll_min_{w}"]  = resampled[col].rolling(w, min_periods=1).min()
            resampled[f"{col}_roll_max_{w}"]  = resampled[col].rolling(w, min_periods=1).max()

    # --- Delta & slope ---
    for col in [c for c in resampled.columns if any(sensor in c for sensor in numeric_cols)]:
        resampled[f"{col}_delta"] = resampled[col].diff()
        resampled[f"{col}_slope"] = resampled[col].diff().rolling(5, min_periods=1).mean()

    # --- FFT features (vibration) ---
    def compute_fft(series, fs=1/60):
        series = series.dropna()
        N = len(series)
        if N < 10: return np.nan
        yf = np.abs(rfft(series))
        xf = rfftfreq(N, fs)
        return xf[np.argmax(yf)]

    vib_cols = ['NDE_motor', 'DE_motor', 'DE_pump', 'NDE_pump', 'Coupling']
    for col in vib_cols:
        colname = col + "_mean"
        if colname in resampled:
            resampled[f"{col}_fft_freq"] = resampled[colname].rolling(60, min_periods=30).apply(compute_fft, raw=False)

    # --- Fill missing ---
    resampled = resampled.fillna(method="bfill").fillna(method="ffill")

    # --- Keep only training features ---
    X_new = resampled[top_features].iloc[[-1]]  # take latest row
    return X_new

# ======= Dashboard page =======
@app.route("/")
def dashboard():
    return render_template("index.html")

# ======= Latest sensor + prediction API =======
@app.route("/latest")
def latest():
    result = {}
    id2label = {0: 'OPERATIONAL', 1: 'OFF', 2: 'CHANGE', 3: 'WARNING', 4: 'CRITICAL'}

    for pump in pumps:
        sensor_dict = generate_sensor_data(pump)
        # Convert dict to DataFrame
        sensor_df = pd.DataFrame([sensor_dict])
        X_new = preprocess_new_data(sensor_df, top_features)

        # Predict next 1-hour status
        pred_class = model.predict(X_new)[0]
        pred_proba = model.predict_proba(X_new)[0].tolist()
        pred_dict = {id2label[i]: round(float(pred_proba[i]), 3) for i in range(len(pred_proba))}

        result[pump] = {
            "status": id2label.get(pred_class, "UNKNOWN"),
            "prediction": pred_dict,
            "sensors": [sensor_dict]
        }


    return jsonify(result)

# ======= Run App =======
if __name__ == "__main__":
    app.run(debug=True)
