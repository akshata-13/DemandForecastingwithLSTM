#  Demand Forecasting with ML and LSTM

This project implements a robust machine learning and deep learning pipeline to forecast product demand using historical sales data.

##  Problem Statement
Given weekly sales data per `store_id` and `sku_id`, the goal is to forecast future `units_sold` values accurately.

##  Dataset
- ~150,000 rows of time-series sales data
- Features: `store_id`, `sku_id`, `week`, `units_sold`, `price`, `total_price`

##  Features Engineered
- Time-based lag features: `units_sold_lag_1`, `lag_2`, `lag_3`
- Grouped time series logic per `(store_id, sku_id)`
- Scaled data using MinMaxScaler for deep learning input

## Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **R² Score**

##  Tech Stack
- Python, Pandas, Scikit-learn
- TensorFlow/Keras for LSTM
- Matplotlib for visualization

##  Results
- Improved MAE from 45.5 (mean-based baseline) to 25.0 using LSTM
- Demonstrated time-series modeling skills and neural network implementation
