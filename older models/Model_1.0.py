#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 12:08:12 2026

@author: sriniramu
Note:
- Implement basic RUL computation from engine max_cycle minus current cycle.
- Select top 13 sensors using absolute correlation with RUL (top_sensors).
- Build linear regression and random forest baselines using top_sensors only.
- Use RUL clipping at 100 cycles to stabilize training targets.
- Train an LSTM model with variable-length (20–50) smart windowing and padding.
- Scale LSTM inputs by flattening across all timesteps and features, then reshaping back.
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import seaborn as sns


index_names = ['engine_id', 'cycle']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = [f'sensor_{i}' for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

folder_path = '/Users/sriniramu/Downloads/Turbofan Engine Project/data' 
train_path = os.path.join(folder_path, 'train_FD001.txt')
test_path  = os.path.join(folder_path, 'test_FD001.txt')
rul_path   = os.path.join(folder_path, 'RUL_FD001.txt')

train_df = pd.read_csv(train_path, sep=r'\s+', header=None, names=col_names)
test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=col_names)
rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])

train_df['engine_id'].nunique()

train_df['max_cycle'] = train_df.groupby('engine_id')['cycle'].transform('max')
train_df['RUL']=train_df['max_cycle']-train_df['cycle']


index = ['engine_id', 'cycle']
settings = ['setting_1', 'setting_2', 'setting_3']
sensors = train_df[sensor_names]


sensor_stats = sensors.agg(['mean', 'std', 'var', 'min', 'max'])
sensor_stats=sensor_stats.transpose()
filter1_sensors = sensor_stats[sensor_stats['var'] == 0] # removing 0 std variables
print(filter1_sensors)
correlation_matrix = train_df.corr()
RUL_correlation = correlation_matrix['RUL'].dropna().sort_values(key=abs,ascending=False)

setting_effect = train_df.groupby('engine_id').agg({
                    'cycle' : 'max',
                    'setting_1' : ['mean'],
                    'setting_2' : ['mean'],
                    'setting_3' : ['mean'],
                    'sensor_1' : ['mean'],
                    'sensor_10' : ['mean'],
                    'sensor_18' : ['mean'],
                    'sensor_19' : ['mean']}).rename(columns={'cycle': 'max_cycle'})
setting_corr = setting_effect.corr()
setting_corr.columns = setting_corr.columns.get_level_values(0)
setting_matrix = setting_corr['max_cycle'].dropna().sort_values(key=abs, ascending=False)
print(setting_matrix)
x=0
top_sensors = RUL_correlation.drop('RUL').head(13).index.tolist()

# fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(8, 6))
# for i in range(6):
#     ax[i,0].scatter(train_df['RUL'], train_df[top_sensors[i+1]], alpha=0.3, s=2)
#     ax[i,0].set_xlabel('RUL'); ax[i,0].set_ylabel(top_sensors[i+1])
#     ax[i,1].scatter(train_df['RUL'], train_df[top_sensors[i+7]], alpha=0.3, s=2)
#     ax[i,1].set_xlabel('RUL'); ax[i,1].set_ylabel(top_sensors[i+7])
    
# plt.tight_layout()
# plt.show()

top_cols = top_sensors + ['RUL']
corr_matrix = train_df[top_cols].corr()

# plt.figure(figsize=(10,8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
#             square=True, fmt='.2f')
# plt.title('Top Sensor Correlations with RUL')
# plt.tight_layout()
# plt.show()

##
model_df1 = train_df[['engine_id', 'RUL'] + top_sensors]

#Model RUL to see where to clip

# X = model_df1[top_sensors]
# y = model_df1['RUL']

# from sklearn.model_selection import train_test_split
# # Get unique engine IDs
# engine_ids = model_df1['engine_id'].unique()

# # Split engines: 80 train, 20 validation
# train_engines, test_engines = train_test_split(engine_ids, test_size=0.2, random_state=0)

# # Filter data by engine
# train_filter = model_df1['engine_id'].isin(train_engines)
# test_filter = model_df1['engine_id'].isin(test_engines)

# X_train, y_train = X[train_filter], y[train_filter]
# X_test, y_test = X[test_filter], y[test_filter]

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# model = LinearRegression()
# model.fit(X_train_scaled, y_train)

# # Predict on validation
# y_pred = model.predict(X_test_scaled)

# # Evaluate
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Validation RMSE: {rmse:.2f} cycles")

# corr_matrix = np.corrcoef(y_test, y_pred)
# print(f"Validation correlation value: {corr_matrix[0, 1]}")

# plt.scatter(y_test, y_pred, alpha=0.3, s=2)
# plt.plot([0,300], [0,300], 'r--')  # Perfect prediction line
# plt.xlabel('True RUL')
# plt.ylabel('Predicted RUL')
# plt.title(f'Baseline Linear Regression | RMSE={rmse:.2f}')
# plt.show()

#Model RUL clipped
RUL_cap = 100
model_df1['RUL_clipped'] = model_df1['RUL'].clip(upper=RUL_cap)

X = model_df1[top_sensors]
y = model_df1['RUL_clipped']

from sklearn.model_selection import train_test_split
# Get unique engine IDs
engines = model_df1['engine_id'].unique()

# Split engines: 80 train, 20 validation
train_engines, test_engines = train_test_split(engines, test_size=0.2, random_state=0)

# Filter data by engine
train_filter = model_df1['engine_id'].isin(train_engines)
test_filter = model_df1['engine_id'].isin(test_engines)

X_train, y_train = X[train_filter], y[train_filter]
X_test, y_test = X[test_filter], y[test_filter]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict using model
y_pred = model.predict(X_test_scaled)

# Evaluate rmse
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Validation RMSE: {rmse:.2f} cycles")

corr_matrix = np.corrcoef(y_test, y_pred)
print(f"Validation correlation value: {corr_matrix[0, 1]}")

# plt.scatter(y_test, y_pred, alpha=0.3, s=2)
# plt.plot([0,RUL_cap], [0,RUL_cap], 'r--')  # Perfect prediction line
# plt.xlabel('True RUL')
# plt.ylabel('Predicted RUL')
# plt.title(f'Baseline Linear Regression | RMSE={rmse:.2f}')
# plt.show()

## RF model (Tree)

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print(f"Random Forest RMSE: {rf_rmse:.2f}")

importances = pd.DataFrame({
    'feature': top_sensors,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print(importances.head(10))

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
#
# # Linear
# ax1.scatter(y_test, model.predict(X_test_scaled), alpha=0.3, s=2)
# ax1.plot([0,RUL_cap], [0,RUL_cap], 'r--'); ax1.set_title(f'Linear\nRMSE={rmse:.2f}')
#
# # Random Forest
# ax2.scatter(y_test, y_pred_rf, alpha=0.3, s=2)
# ax2.plot([0,RUL_cap], [0,RUL_cap], 'r--'); ax2.set_title(f'Random Forest\nRMSE={rf_rmse:.2f}')
#
# plt.tight_layout()
# plt.show()

## LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

seq_len_min = 20
seq_len_max = 50
X_seq, y_seq, seq_engine_ids = [], [], []
for engine in engines:
    eng_data = model_df1[model_df1['engine_id'] == engine].sort_values('cycle')
    for i in range(seq_len_min, len(eng_data)): # 31
        seq_size = min(i, seq_len_max) # seq_size = 31
        X_seq.append(eng_data.iloc[i - seq_size:i][top_sensors].values) # [0:31][
        y_seq.append(eng_data.iloc[i]['RUL_clipped'])
        seq_engine_ids.append(engine)


y_seq = np.array(y_seq)
seq_engine_ids = np.array(seq_engine_ids)

from tensorflow.keras.preprocessing.sequence import pad_sequences

X_seq_padded = pad_sequences(X_seq, maxlen=50, padding='pre', dtype='float32')

train_mask = np.isin(seq_engine_ids, train_engines)
test_mask = np.isin(seq_engine_ids, test_engines)

X_train_L, y_train_L = X_seq_padded[train_mask], y_seq[train_mask]
X_test_L, y_test_L = X_seq_padded[test_mask], y_seq[test_mask]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_flat = X_train_L.reshape(-1, X_train_L.shape[2])
X_train_scaled_L = scaler.fit_transform(X_train_flat).reshape(X_train_L.shape)

X_test_flat = X_test_L.reshape(-1, X_train_L.shape[2])
X_test_scaled_L = scaler.transform(X_test_flat).reshape(X_test_L.shape)

# Reshape for LSTM: add feature dimension if needed
# X_seq already (samples, timesteps, features)

model_lstm = Sequential([
    tf.keras.layers.Masking(mask_value=0.0, input_shape=(seq_len_max, X_seq_padded.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(.2),
    LSTM(64 // 2),
    Dropout(.2),
    Dense(32, activation='relu'),
    Dense(1)])

model_lstm.compile(optimizer='adam', loss='mse',metrics=[])
history = model_lstm.fit(
    X_train_scaled_L, y_train_L,           # Your sequences
    validation_data=(X_test_scaled_L, y_test_L),
    epochs=30,                          # Train 50 passes through data
    batch_size=64,                      # Process 64 sequences at once
    verbose=1
)



# RMSE
y_pred_L = model_lstm.predict(X_test_scaled_L).ravel()
lstm_rmse = np.sqrt(mean_squared_error(y_test_L, y_pred_L))
print(f"LSTM RMSE: {lstm_rmse:.2f} cycles")
print(f"vs RF baseline: {rf_rmse:.2f} cycles")

# # Plot predictions
# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.scatter(y_test_L, y_pred_L, alpha=0.3, s=2)
# plt.plot([0,RUL_cap],[0,RUL_cap],'r--')
# plt.xlabel('True RUL'); plt.ylabel('Pred RUL')
# plt.title(f'LSTM | RMSE={lstm_rmse:.1f}')
#
# plt.subplot(1,2,2)
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='val')
# plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()
# plt.title('Training curves')
#
# plt.tight_layout()
# plt.show()
#

# variables = np.array([
#     [1e-4, 3e-4, 5e-4, 1e-3],   # row 0: lr
#     [16,   32,   64,   128],    # row 1: batch_size
#     [0.0,  0.1,  0.2,  0.3],    # row 2: dropout
#     [16,   32,   64,   64],     # row 3: dense_units
#     [32,   64,   128,  256]     # row 4: lstm_units
# ], dtype=object)
# best_val = np.array([1e-3, 32, 0.2, 32, 64], dtype=object)
# param_names = ['lr', 'batch_size', 'dropout', 'dense_units', 'lstm_units']
# best_rmse = 12.00
# for i in range (len(best_val)):
#         for j in range (4):
#             config = best_val.copy()
#             config[i] = variables[i, j]
#             model = Sequential([
#                 tf.keras.layers.Masking(mask_value=0.0, input_shape=(seq_len, X_seq_padded.shape[2])),
#                 LSTM(64, return_sequences=True),
#                 Dropout(config[2]),
#                 LSTM(config[4] // 2),
#                 Dropout(config[2]),
#                 Dense(config[3], activation='relu'),
#                 Dense(1)
#             ])
#             model.compile(
#                 optimizer=Adam(learning_rate=config[0]),
#                 loss='mse'
#             )
#             history = model.fit(
#                 X_train_scaled_L, y_train_L,
#                 validation_data=(X_test_scaled_L, y_test_L),
#                 epochs=30,
#                 batch_size = config[1],
#                 verbose=1)
#             y_pred_L = model.predict(X_test_scaled_L, verbose=0).ravel()
#             rmse = np.sqrt(mean_squared_error(y_test_L, y_pred_L))
#             if rmse < best_rmse:
#                 best_rmse = rmse
#                 best_val[i] = variables[i, j]
#
#         print(f"✅ Best {param_names[i]}: {best_val[i]} (RMSE {best_rmse:.2f})")
#



