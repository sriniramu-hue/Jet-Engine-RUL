# Turbofan RUL Prediction (NASA C-MAPSS FD001)

Predicts remaining useful life of turbofan engines using physics-informed ML.

## Results
| Model | Val RMSE | Test RMSE |
|-------|----------|-----------|
| Log-Linear | 17.36 | 17.82 |
| Random Forest | 15.04 | 15.30 |
| LSTM (30-cycle) | 13.78 | **13.10** |

## Key Innovations
- **Physics features**: HPC efficiency (T30/T24), pressure ratio (P30/P2), bleed flow proxies.
- **ElasticNetCV + GroupKFold**: Engine-level feature selection.
- **Fixed 30-cycle LSTM**: Temporal degradation modeling.

## Usage
```bash
pip install -r requirements.txt
python models/v4_final.py
