pip install pandas matplotlib scikit-learn statsmodels
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("/content/Sample - Superstore.csv", encoding='latin1')
df['Order Date'] = pd.to_datetime(df['Order Date'])
monthly_sales = df.groupby(
    pd.Grouper(key='Order Date', freq='M')
)['Sales'].sum()
plt.figure(figsize=(12,6))
plt.plot(monthly_sales)
plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid()
plt.show()
train = monthly_sales[:-12]
test = monthly_sales[-12:]
model = ARIMA(train, order=(5,1,0))

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test, forecast)
accuracy = (1 - (mae / test.mean())) * 100
print("Forecast Accuracy:", accuracy)