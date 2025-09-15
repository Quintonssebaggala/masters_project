import pandas as pd
import numpy as np
from scipy.fft import rfft, rfftfreq

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

