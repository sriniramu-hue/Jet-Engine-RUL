"""
Update:
- Add physics-based features: HPC_efficiency (sensor_11/sensor_2), pressure_ratio (sensor_7/sensor_5),
  sensor_11_dev (deviation from engine's first 10 cycles).
- Use full feature set (settings + all sensors + physics features) for ElasticNetCV.
- Implement log-RUL transformation (clip to avoid log(0), exp back for RMSE).
- Use GroupKFold by engine_id for leakage-free cross-validation.
- Fit ElasticNetCV pipeline (StandardScaler + model) on log-RUL train folds only.
- Print per-fold RMSE and selected features (non-zero coefficients) to identify key sensors/physics proxies.
- Report mean ± std RMSE across 5 folds for the exponential ElasticNet model.
"""


from enum import auto

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
train_df['HPC_efficiency'] = train_df['sensor_11'] / train_df['sensor_2']  # T30/T24
train_df['pressure_ratio'] = train_df['sensor_7'] / train_df['sensor_5']  # P30/P2
train_df['sensor_11_dev'] = train_df['sensor_11'] / train_df.groupby('engine_id')['sensor_11'].transform(lambda x: x.iloc[:10].mean())



index = ['engine_id', 'cycle']
settings = ['setting_1', 'setting_2', 'setting_3']
sensors = train_df[sensor_names]

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

# ----- Data prep -----
# X = all features except RUL, max_cycle
X = train_df.drop(columns=['RUL', 'max_cycle'])
y = train_df['RUL'].values
groups = train_df['engine_id'].values

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
    print("  Selected sensors:", list(selected))

print(f"\nElasticNetCV exponential model RMSE: {np.mean(rmse_folds):.2f} ± {np.std(rmse_folds):.2f}")

# best_alpha = pipe.named_steps["model"].alpha_
# best_l1_ratio = pipe.named_steps["model"].l1_ratio_
# print(best_alpha)
# print(best_l1_ratio)

# model = svm.SVC(kernel='rbf', C=10, gamma='auto')
#
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# skf = StratifiedKFold(n_splits=5, shuffle= True, random_state= 42)
# val_scores = cross_val_score(estimator= model,  X= X_train, y= y_train, cv= skf)

#use random search due to large num of parameters