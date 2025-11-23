# Creating-a-Project-for-forecasting-using-ARIMA-model-in-Python

# Name : Moenish Baalan G
# Reg No : 212223220057

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.

### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
8. Evaluate model predictions

### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("results.csv")

data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(by='date')

yearly_data = data.groupby(data['date'].dt.year)['home_score'].mean().reset_index()
yearly_data.rename(columns={'date': 'Year', 'home_score': 'Avg_Home_Score'}, inplace=True)

yearly_data.set_index('Year', inplace=True)

target_variable = 'Avg_Home_Score'

def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test_data))
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data', color='blue')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data', color='green')
    plt.plot(test_data.index, forecast, label='Forecasted Data', linestyle='--', color='red')
    plt.xlabel('Year')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting of Avg Home Score')
    plt.legend()
    plt.grid()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

arima_model(yearly_data, target_variable, order=(2, 1, 1))

```

### OUTPUT:
<img width="1291" height="677" alt="image" src="https://github.com/user-attachments/assets/802c69ba-2538-4cdb-87a4-7332f3a5229d" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
