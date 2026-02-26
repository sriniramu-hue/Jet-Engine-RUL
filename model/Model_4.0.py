#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version 4:
Update:
- Add physics-based features (HPC_efficiency, pressure_ratio, sensor_11_dev, bleed_total,
  bleed_ratio_core, core_flow_proxy) to the training dataframe.
- Keep correlation-based top_sensors logic identical to the original baseline.
- Form base_feature_cols by combining top_sensors with the physics features.
- Use ElasticNetCV with GroupKFold over engine_id on base_feature_cols to select
  a sparse, physics-aware feature subset.
- Print selected features for each fold plus the final ElasticNet-selected feature list
  so physics-based parameters can be inspected.
- Re-train the random forest model using the ElasticNet-selected features only.
- Build LSTM inputs from the ElasticNet-selected features and use a fixed-length
  30-cycle sliding window (no smart windowing or padding).
- Scale LSTM inputs by flattening across all timesteps and features, then reshaping back,
  and train a 2-layer LSTM (64, 32 units) with dropout and a dense head to predict RUL_clipped.
- ADD TEST SET EVALUATION: Load test_FD001.txt + RUL_FD001.txt, compute RUL_true using
  canonical offset method, build last-cycle sequences from selected_features, predict,
  and report test RMSE with scatter plot.
- CHANGE LINEAR MODEL: log-linear regression (log target, exp back) to match
  exponential degradation shape instead of naive linear fit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ----------------------------
# Load data
# ----------------------------

index_names = ['engine_id', 'cycle']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = [f'sensor_{i}' for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(ROOT_DIR, "data")
train_path = os.path.join(data_dir, 'train_FD001.txt')
test_path  = os.path.join(data_dir, 'test_FD001.txt')
rul_path   = os.path.join(data_dir, 'RUL_FD001.txt')

train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)

# ----------------------------
# RUL for train
# ----------------------------

train_df['max_cycle'] = train_df.groupby('engine_id')['cycle'].transform('max')
train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']

sensors = train_df[sensor_names]

# ----------------------------
# Physics-based features (train)
# ----------------------------

eps = 1e-6

train_df['HPC_efficiency'] = train_df['sensor_11'] / (train_df['sensor_2'] + eps)
train_df['pressure_ratio'] = train_df['sensor_7']  / (train_df['sensor_5'] + eps)

baseline_11 = train_df.groupby('engine_id')['sensor_11'].transform(lambda x: x.iloc[:10].mean())
train_df['sensor_11_dev']  = train_df['sensor_11'] / (baseline_11 + eps)

train_df['bleed_total'] = train_df['sensor_20'] + train_df['sensor_21']
train_df['bleed_ratio_core'] = train_df['bleed_total'] / (train_df['sensor_9'] + eps)
train_df['core_flow_proxy'] = (train_df['sensor_9'] * train_df['sensor_7'] / np.sqrt(train_df['sensor_3'] + eps)
)

physics_features = [
    'HPC_efficiency',
    'pressure_ratio',
    'sensor_11_dev',
    'bleed_total',
    'bleed_ratio_core',
    'core_flow_proxy'
]

# ----------------------------
# Sensor stats, filter zero-variance sensors
# ----------------------------

sensor_stats = sensors.agg(['mean', 'std', 'var', 'min', 'max']).transpose()
filter1_sensors = sensor_stats[sensor_stats['var'] == 0]
print("Zero-variance sensors (to ignore):")
print(filter1_sensors)

# ----------------------------
# Correlation-based top sensors
# ----------------------------

correlation_matrix = train_df.corr()
RUL_correlation = correlation_matrix['RUL'].dropna().sort_values(key=abs, ascending=False)

setting_effect = train_df.groupby('engine_id').agg({
    'cycle': 'max',
    'setting_1':  ['mean'],
    'setting_2':  ['mean'],
    'setting_3':  ['mean'],
    'sensor_1':  ['mean'],
    'sensor_10':  ['mean'],
    'sensor_18':  ['mean'],
    'sensor_19':  ['mean']
}).rename(columns={'cycle': 'max_cycle'})
setting_corr = setting_effect.corr()
setting_corr.columns = setting_corr.columns.get_level_values(0)
setting_matrix = setting_corr['max_cycle'].dropna().sort_values(key=abs, ascending=False)
print("Setting correlation with max_cycle:")
print(setting_matrix)

top_sensors = RUL_correlation.drop('RUL').head(13).index.tolist()
print("Top sensors by abs(corr) with RUL:")
print(top_sensors)

# ----------------------------
# Base modeling dataframe (train)
# ----------------------------

RUL_cap = 125
train_df['RUL_clipped'] = train_df['RUL'].clip(upper=RUL_cap)

base_feature_cols = list(dict.fromkeys(top_sensors + physics_features))
model_df1 = train_df[['engine_id', 'RUL', 'RUL_clipped'] + base_feature_cols]

engines = model_df1['engine_id'].unique()
train_engines, val_engines = train_test_split(engines, test_size=0.2, random_state=0)
train_filter = model_df1['engine_id'].isin(train_engines)
val_filter = model_df1['engine_id'].isin(val_engines)

# ----------------------------
# Log-Linear Model
# ----------------------------

X_lin = model_df1[top_sensors]
y_lin = model_df1['RUL_clipped']

# Log-transform features (X), keep y on linear scale
X_lin_log = np.log(np.clip(X_lin, 1e-3, None))

X_train_lin_log, y_train_lin = X_lin_log[train_filter], y_lin[train_filter]
X_val_lin_log,   y_val_lin   = X_lin_log[val_filter],   y_lin[val_filter]

scaler_lin = MinMaxScaler()
X_train_lin_log_scaled = scaler_lin.fit_transform(X_train_lin_log)
X_val_lin_log_scaled = scaler_lin.transform(X_val_lin_log)

lin_model = LinearRegression()
lin_model.fit(X_train_lin_log_scaled, y_train_lin)   # fit on log-X, raw y

y_pred_lin = lin_model.predict(X_val_lin_log_scaled) # predict raw y directly, no exp
rmse_lin = np.sqrt(mean_squared_error(y_val_lin, y_pred_lin))
print(f"\nLog-X Linear Regression Val RMSE: {rmse_lin:.2f} cycles")

# ----------------------------
# ElasticNetCV + GroupKFold for feature selection (including physics)
# ----------------------------

X_enet = model_df1[base_feature_cols]
y_enet = model_df1['RUL_clipped'].values
groups = model_df1['engine_id'].values

gkf = GroupKFold(n_splits=5)

enet_model = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.9],
    alphas=np.logspace(-3, 1, 20),
    cv=5,
    max_iter=10000
)

rmse_folds = []
selected_features_all_folds = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_enet, y_enet, groups=groups), start=1):
    X_tr, X_val = X_enet.iloc[tr_idx], X_enet.iloc[val_idx]
    y_tr, y_val = y_enet[tr_idx], y_enet[val_idx]

    y_tr_log = np.log(np.clip(y_tr, 1e-3, None))
    enet_model.fit(X_tr, y_tr_log)

    y_val_pred = np.exp(enet_model.predict(X_val))
    rmse_fold = np.sqrt(mean_squared_error(y_val, y_val_pred))
    rmse_folds.append(rmse_fold)

    coef = enet_model.coef_
    selected = X_enet.columns[coef != 0]
    selected_features_all_folds.append(list(selected))

    print(f"Fold {fold} RMSE (ElasticNet exp): {rmse_fold:.2f}")
    print("  Selected features:", list(selected))

print(f"\nElasticNet exponential model RMSE (mean ± std): "
      f"{np.mean(rmse_folds):.2f} ± {np.std(rmse_folds):.2f}")

# Refit on all data for final feature selection
enet_model.fit(X_enet, np.log(np.clip(y_enet, 1e-3, None)))
selected_features = X_enet.columns[enet_model.coef_ != 0].tolist()

print("\nFinal selected features from ElasticNet (including physics):")
for f in selected_features:
    print(" -", f)

# ----------------------------
# Random Forest (ElasticNet-selected features)
# ----------------------------

X_rf = model_df1[selected_features]
y_rf = model_df1['RUL_clipped']

X_train_rf, y_train_rf = X_rf[train_filter], y_rf[train_filter]
X_val_rf,   y_val_rf   = X_rf[val_filter],   y_rf[val_filter]

scaler_rf = MinMaxScaler()
X_train_rf_scaled = scaler_rf.fit_transform(X_train_rf)
X_val_rf_scaled = scaler_rf.transform(X_val_rf)

rf_model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
rf_model.fit(X_train_rf_scaled, y_train_rf)
y_pred_rf = rf_model.predict(X_val_rf_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_val_rf, y_pred_rf))
print(f"\nRandom Forest Val RMSE: {rf_rmse:.2f}")

importances = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nRandom Forest feature importances (top 10):")
print(importances.head(10))

# ----------------------------
# Load & prep test data (shared for all models)
# ----------------------------

test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=col_names)
rul_df = pd.read_csv(rul_path,  sep=r'\s+', header=None, names=['RUL_last'])

max_cycle_test = test_df.groupby('engine_id')['cycle'].max().reset_index()
max_cycle_test.columns = ['engine_id', 'max_cycle_test']
max_cycle_test['RUL_last'] = rul_df['RUL_last'].values
max_cycle_test['offset'] = max_cycle_test['max_cycle_test'] + max_cycle_test['RUL_last']

offset_map = dict(zip(max_cycle_test['engine_id'], max_cycle_test['offset']))
test_df['RUL_true'] = test_df.apply(lambda row: offset_map[row['engine_id']] - row['cycle'], axis=1)
test_df['RUL_true_clipped'] = test_df['RUL_true'].clip(upper=RUL_cap)

# Apply same physics features to test
test_df['HPC_efficiency'] = test_df['sensor_11'] / (test_df['sensor_2'] + eps)
test_df['pressure_ratio'] = test_df['sensor_7']  / (test_df['sensor_5'] + eps)

baseline_11_test = test_df.groupby('engine_id')['sensor_11'].transform(lambda x: x.iloc[:10].mean())
test_df['sensor_11_dev'] = test_df['sensor_11'] / (baseline_11_test + eps)

test_df['bleed_total'] = test_df['sensor_20'] + test_df['sensor_21']
test_df['bleed_ratio_core'] = test_df['bleed_total'] / (test_df['sensor_9'] + eps)
test_df['core_flow_proxy'] = (test_df['sensor_9'] * test_df['sensor_7'] / np.sqrt(test_df['sensor_3'] + eps)
)

# Last observed cycle per engine (tabular model benchmark protocol)
test_last = test_df.loc[test_df.groupby('engine_id')['cycle'].idxmax()].reset_index(drop=True)

# ----------------------------
# Log-Linear test evaluation
# ----------------------------

X_test_lin_final = test_last[top_sensors]
y_test_lin_final = test_last['RUL_true_clipped']

X_test_lin_log = np.log(np.clip(X_test_lin_final, 1e-3, None))
X_test_lin_log_scaled = scaler_lin.transform(X_test_lin_log)  # use train scaler

y_pred_lin_final = lin_model.predict(X_test_lin_log_scaled)   # no exp
rmse_lin_test = np.sqrt(mean_squared_error(y_test_lin_final, y_pred_lin_final))
print(f"\nLog-X Linear Test RMSE: {rmse_lin_test:.2f} cycles")

# ----------------------------
# Random Forest test evaluation
# ----------------------------

X_test_rf_final = test_last[selected_features]
y_test_rf_final = test_last['RUL_true_clipped']
X_test_rf_final_scaled = scaler_rf.transform(X_test_rf_final)
y_pred_rf_final = rf_model.predict(X_test_rf_final_scaled)

rmse_rf_test = np.sqrt(mean_squared_error(y_test_rf_final, y_pred_rf_final))
print(f"Random Forest Test RMSE: {rmse_rf_test:.2f} cycles")

plt.figure(figsize=(7, 6))
plt.scatter(y_test_rf_final, y_pred_rf_final, alpha=0.7, edgecolors='k', linewidths=0.4)
plt.plot([0, RUL_cap], [0, RUL_cap], 'r--', label='Perfect prediction')
plt.xlabel('True RUL (cycles)')
plt.ylabel('Predicted RUL (cycles)')
plt.title(f'Random Forest Test Set | FD001 | RMSE = {rmse_rf_test:.2f} cycles')
plt.legend()
plt.tight_layout()
plt.show()

rf_results_df = pd.DataFrame({
    'engine_id': test_last['engine_id'].values,
    'true_RUL': y_test_rf_final.values,
    'pred_RUL': y_pred_rf_final,
    'error': y_pred_rf_final - y_test_rf_final.values
}).sort_values('engine_id').reset_index(drop=True)

print("\nRF Per-engine prediction sample (first 10):")
print(rf_results_df.head(10).to_string(index=False))

plt.figure(figsize=(12, 4))
plt.bar(rf_results_df['engine_id'], rf_results_df['error'], color='darkorange', alpha=0.7)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Engine ID')
plt.ylabel('Prediction Error (pred - true) cycles')
plt.title('Per-Engine RUL Prediction Error | Random Forest | FD001 Test Set')
plt.tight_layout()
plt.show()

# ----------------------------
# LSTM with fixed-length window (seq_len=30)
# ----------------------------

seq_len = 30
feature_cols_lstm = selected_features

X_seq, y_seq, seq_engine_ids = [], [], []

for engine in engines:
    eng_data = model_df1[model_df1['engine_id'] == engine].sort_values('cycle').reset_index(drop=True)
    for i in range(seq_len, len(eng_data)):
        X_seq.append(eng_data.iloc[i-seq_len:i][feature_cols_lstm].values)
        y_seq.append(eng_data.iloc[i]['RUL_clipped'])
        seq_engine_ids.append(engine)

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)
seq_engine_ids = np.array(seq_engine_ids)

train_mask_L = np.isin(seq_engine_ids, train_engines)
val_mask_L = np.isin(seq_engine_ids, val_engines)

X_train_L, y_train_L = X_seq[train_mask_L], y_seq[train_mask_L]
X_val_L,   y_val_L   = X_seq[val_mask_L],   y_seq[val_mask_L]

scaler_L = StandardScaler()
n_features_L = X_train_L.shape[2]

X_train_scaled_L = scaler_L.fit_transform(X_train_L.reshape(-1, n_features_L)).reshape(X_train_L.shape)
X_val_scaled_L = scaler_L.transform(X_val_L.reshape(-1, n_features_L)).reshape(X_val_L.shape)

model_lstm = Sequential([
    LSTM(64, input_shape=(seq_len, n_features_L), return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mse')

history = model_lstm.fit(
    X_train_scaled_L, y_train_L,
    validation_data=(X_val_scaled_L, y_val_L),
    epochs=30,
    batch_size=64,
    verbose=1
)

y_pred_L = model_lstm.predict(X_val_scaled_L).ravel()
lstm_rmse = np.sqrt(mean_squared_error(y_val_L, y_pred_L))
print(f"\nLSTM Val RMSE: {lstm_rmse:.2f} cycles")

# ----------------------------
# LSTM test evaluation
# ----------------------------

X_seq_test, y_seq_test, test_engine_ids = [], [], []

for engine in test_df['engine_id'].unique():
    eng_data = test_df[test_df['engine_id'] == engine].sort_values('cycle').reset_index(drop=True)
    if len(eng_data) < seq_len:
        continue
    last_i = len(eng_data)
    X_seq_test.append(eng_data.iloc[last_i - seq_len:last_i][feature_cols_lstm].values)
    y_seq_test.append(eng_data.iloc[last_i - 1]['RUL_true_clipped'])
    test_engine_ids.append(engine)

X_seq_test = np.array(X_seq_test, dtype=np.float32)
y_seq_test = np.array(y_seq_test, dtype=np.float32)

X_test_scaled_final = scaler_L.transform(X_seq_test.reshape(-1, n_features_L)).reshape(X_seq_test.shape)

y_test_pred_final = model_lstm.predict(X_test_scaled_final).ravel()
rmse_test_final = np.sqrt(mean_squared_error(y_seq_test, y_test_pred_final))

print(f"\n========== FINAL TEST SET RESULTS ==========")
print(f"Log-Linear Test RMSE:    {rmse_lin_test:.2f} cycles")
print(f"Random Forest Test RMSE: {rmse_rf_test:.2f} cycles")
print(f"LSTM Test RMSE:          {rmse_test_final:.2f} cycles")
print(f"--------------------------------------------")
print(f"Log-Linear Val RMSE:     {rmse_lin:.2f} cycles")
print(f"Random Forest Val RMSE:  {rf_rmse:.2f} cycles")
print(f"LSTM Val RMSE:           {lstm_rmse:.2f} cycles")

plt.figure(figsize=(7, 6))
plt.scatter(y_seq_test, y_test_pred_final, alpha=0.7, edgecolors='k', linewidths=0.4)
plt.plot([0, RUL_cap], [0, RUL_cap], 'r--', label='Perfect prediction')
plt.xlabel('True RUL (cycles)')
plt.ylabel('Predicted RUL (cycles)')
plt.title(f'LSTM Test Set | FD001 | RMSE = {rmse_test_final:.2f} cycles')
plt.legend()
plt.tight_layout()
plt.show()

results_df = pd.DataFrame({
    'engine_id': test_engine_ids,
    'true_RUL':  y_seq_test,
    'pred_RUL':  y_test_pred_final,
    'error':  y_test_pred_final - y_seq_test
}).sort_values('engine_id').reset_index(drop=True)

print("\nLSTM Per-engine prediction sample (first 10):")
print(results_df.head(10).to_string(index=False))

plt.figure(figsize=(12, 4))
plt.bar(results_df['engine_id'], results_df['error'], color='steelblue', alpha=0.7)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Engine ID')
plt.ylabel('Prediction Error (pred - true) cycles')
plt.title('Per-Engine RUL Prediction Error | LSTM | FD001 Test Set')
plt.tight_layout()
plt.show()
