
"""
Update:
- Add physics-based features: HPC_efficiency (sensor_11/sensor_2), pressure_ratio (sensor_7/sensor_5),
  sensor_11_dev (per-engine early-life baseline).
- Form X as all settings + sensors + physics features (drop RUL, max_cycle only).
- Use ElasticNetCV Pipeline with log-RUL + GroupKFold(5) to select sparse features.
- Print per-fold selected features and final selected_features list after refit on all data.
- Build fixed-length sequences (seq_len=30) from selected_features only.
- Split sequences by engine_id (train_engines/val_engines, random_state=0).
- Scale LSTM inputs via flatten → StandardScaler → reshape across all timesteps/features.
- Train 2-layer LSTM (64→32 units, dropout 0.2, Adam lr=1e-3) on sequences to predict raw RUL.
- Report validation RMSE on LSTM predictions.
"""


# ================== IMPORTS ==================
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ================== LOAD DATA ==================
index_names = ['engine_id', 'cycle']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = [f'sensor_{i}' for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

folder_path = '/Users/sriniramu/Downloads/Turbofan Engine Project/data'
train_path = os.path.join(folder_path, 'train_FD001.txt')
test_path = os.path.join(folder_path, 'test_FD001.txt')
rul_path = os.path.join(folder_path, 'RUL_FD001.txt')

train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)

# ================== RUL + PHYSICS FEATURES ==================
# Compute RUL
train_df['max_cycle'] = train_df.groupby('engine_id')['cycle'].transform('max')
train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']

# Example physics-informed features
train_df['HPC_efficiency'] = train_df['sensor_11'] / train_df['sensor_2']  # T30/T24 proxy
train_df['pressure_ratio'] = train_df['sensor_7'] / train_df['sensor_5']  # P30/P2 proxy
train_df['sensor_11_dev'] = train_df['sensor_11'] / train_df.groupby('engine_id')['sensor_11'].transform(
    lambda x: x.iloc[:10].mean())

# Features matrix X: drop target and helper
X = train_df.drop(columns=['RUL', 'max_cycle'])
y = train_df['RUL'].values
groups = train_df['engine_id'].values

# ================== ELASTICNETCV (LOG-RUL) + GROUPKFOLD ==================
enet_log = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=np.logspace(-3, 1, 20),
        cv=5,
        max_iter=10000
    ))
])

gkf = GroupKFold(n_splits=5)
rmse_folds = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    # log-transform RUL (avoid log(0))
    y_tr_pos = np.clip(y_tr, 1e-3, None)
    y_tr_log = np.log(y_tr_pos)

    # fit on TRAIN ONLY
    enet_log.fit(X_tr, y_tr_log)

    # predict on VAL, then exp back to RUL
    y_val_log_pred = enet_log.predict(X_val)
    y_val_pred = np.exp(y_val_log_pred)

    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    rmse_folds.append(rmse)

    coef = enet_log.named_steps["model"].coef_
    selected = X.columns[coef != 0]

    print(f"Fold {fold} RMSE: {rmse:.2f}")
    print("  Selected features:", list(selected))

print(f"\nElasticNetCV exponential model RMSE: {np.mean(rmse_folds):.2f} ± {np.std(rmse_folds):.2f}")

# ================== FINAL SENSOR SELECTION (FIT ON ALL DATA) ==================
y_pos_all = np.clip(y, 1e-3, None)
y_log_all = np.log(y_pos_all)
enet_log.fit(X, y_log_all)

coef_all = enet_log.named_steps["model"].coef_
selected_mask = coef_all != 0
selected_features = X.columns[selected_mask]

print("\nFinal selected features for LSTM:")
print(list(selected_features))

# ================== BUILD SEQUENCES FROM SELECTED FEATURES ==================
seq_len = 30
X_seq, y_seq, seq_engine_ids = [], [], []

for engine in train_df['engine_id'].unique():
    eng_data = train_df[train_df['engine_id'] == engine].sort_values('cycle').reset_index(drop=True)

    for i in range(seq_len, len(eng_data)):
        X_seq.append(eng_data.iloc[i - seq_len:i][selected_features].values)
        y_seq.append(eng_data.iloc[i]['RUL'])  # or RUL_clipped if you choose to clip
        seq_engine_ids.append(engine)

X_seq = np.array(X_seq)  # (n_seq, seq_len, n_features_sel)
y_seq = np.array(y_seq)
seq_engine_ids = np.array(seq_engine_ids)
n_features = X_seq.shape[2]

print(f"\nSequence tensor: {X_seq.shape}, labels: {y_seq.shape}")

# ================== TRAIN/VAL SPLIT BY ENGINE ==================
engines = train_df['engine_id'].unique()
train_engines, val_engines = train_test_split(engines, test_size=0.2, random_state=0)

train_mask = np.isin(seq_engine_ids, train_engines)
val_mask = np.isin(seq_engine_ids, val_engines)

X_train_L, y_train_L = X_seq[train_mask], y_seq[train_mask]
X_val_L, y_val_L = X_seq[val_mask], y_seq[val_mask]

# ================== SCALE FEATURES FOR LSTM ==================
scaler_seq = StandardScaler()
X_train_flat = X_train_L.reshape(-1, n_features)
X_train_scaled = scaler_seq.fit_transform(X_train_flat).reshape(X_train_L.shape)

X_val_flat = X_val_L.reshape(-1, n_features)
X_val_scaled = scaler_seq.transform(X_val_flat).reshape(X_val_L.shape)

# ================== DEFINE & TRAIN LSTM ==================
model_lstm = Sequential([
    LSTM(64, input_shape=(seq_len, n_features), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model_lstm.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

history = model_lstm.fit(
    X_train_scaled, y_train_L,
    validation_data=(X_val_scaled, y_val_L),
    epochs=50,
    batch_size=32,
    verbose=1
)

# ================== EVALUATE LSTM (RMSE) ==================
y_val_pred = model_lstm.predict(X_val_scaled).ravel()
mse_lstm = mean_squared_error(y_val_L, y_val_pred)
rmse_lstm = np.sqrt(mse_lstm)
print(f"\nLSTM (ElasticNet-selected features) val RMSE: {rmse_lstm:.2f} cycles")
