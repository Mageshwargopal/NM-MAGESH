# NM-MAGESH
Future Sales Prediction Project

Overview
This project aims to build a machine learning model for predicting future sales based on historical data. Both traditional statistical methods like ARIMA and advanced machine learning algorithms like XGBoost are explored.

Dependencies
Python 3.x
Pandas
Numpy
Scikit-learn
Matplotlib
TensorFlow
Keras
Statsmodels

Data
The data used is monthly retail sales data with the following features:

Date
Sales Amount
Marketing Expenses
Number of Products
Competitor Actions

Files
data_loading.py - Script for loading and preprocessing data
eda.py - Exploratory data analysis visualizations
feature_engineering.py - Script for feature engineering steps
models.py - Code for training and evaluating machine learning models
model_selection.py - Model selection based on performance metrics
arima.py - ARIMA time series model building
lstm.py - LSTM neural network model
model_results.csv - Performance results for comparison

Usage
The modeling pipeline can be run end-to-end starting with data loading and preprocessing in data_loading.py followed by individual model scripts. The final model selection is done in model_selection.py based on results in model_results.csv.

Results
The XGBoost model performed best with lowest MAE and RMSE on test data. The ARIMA model also produced decent forecasts indicating sales time series has seasonal patterns. The LSTM neural network model overfit on training data indicating more tuning is required.

Future Work
Try more advanced feature engineering and external datasets
Ensemble modeling with XGBoost and ARIMA
Hyperparameter tuning for LSTM model
Deploy top model to production environment

References
Brownlee, J. (2020). Time Series Forecasting With Prophet in Python. Machine Learning Mastery.
Brownlee, J. (2020). Multivariate Time Series Forecasting With LSTMs in Keras. Machine Learning Mastery.
Hyndman, R.J. and Athanasopoulos, G. (2021). Forecasting: principles and practice. OTexts.
Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting.
Montero-Manso, P., Athanasopoulos, G., Hyndman, R.J. and Talagala, T.S. (2020). FFORMA: Feature-based Forecasting Model Averaging. International Journal of Forecasting.
Siami-Namini, S., Tavakoli, N. and Namin, A.S. (2018). A Comparison of ARIMA and LSTM in Forecasting Time Series. 17th IEEE International Conference on Machine Learning and Applications (ICMLA).
Tseng, F.M., Yu, H.C. and Tzeng, G.H. (2002). Combining neural network model with seasonal time series ARIMA model. Technological Forecasting and Social Change.
Zhang, G.P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. Neurocomputing.
