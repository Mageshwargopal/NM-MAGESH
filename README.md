# NM-MAGESH
**Future Sales Prediction Project**

*Overview*

This project aims to build a machine learning model for predicting future sales based on historical data. Both traditional statistical methods like ARIMA and advanced machine learning algorithms like XGBoost are explored.

*Dependencies*

1)Python 3.x
2)Pandas
3)Numpy
4)Scikit-learn
5)Matplotlib
6)TensorFlow
7)Keras
8)Statsmodels

*Data*

The data used is monthly retail sales data with the following features:

1)Date
2)Sales Amount
3)Marketing Expenses
4)Number of Products
5)Competitor Actions

*Files*

1)data_loading.py - Script for loading and preprocessing data
2)eda.py - Exploratory data analysis visualizations
3)feature_engineering.py - Script for feature engineering steps
4)models.py - Code for training and evaluating machine learning models
5)model_selection.py - Model selection based on performance metrics
6)arima.py - ARIMA time series model building
7)lstm.py - LSTM neural network model
8)model_results.csv - Performance results for comparison

*Usage*

The modeling pipeline can be run end-to-end starting with data loading and preprocessing in data_loading.py followed by individual model scripts. The final model selection is done in model_selection.py based on results in model_results.csv.

*Results*

The XGBoost model performed best with lowest MAE and RMSE on test data. The ARIMA model also produced decent forecasts indicating sales time series has seasonal patterns. The LSTM neural network model overfit on training data indicating more tuning is required.

*Future Work*

1)Try more advanced feature engineering and external datasets
2)Ensemble modeling with XGBoost and ARIMA
3)Hyperparameter tuning for LSTM model
4)Deploy top model to production environment

