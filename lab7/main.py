import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Отримати дані про ціни акцій компанії Apple (AAPL)
data = yf.download('AAPL', start='2020-01-01', end='2024-04-30')
# Вивести перші кілька рядків даних для перевірки
print(data.head())

# Аналіз даних

data['Adj Close'].plot(figsize=(10, 6))
plt.title('Ціни акцій компанії Apple (AAPL)')
plt.ylabel('Ціна')
plt.xlabel('Дата')
plt.grid(True)
plt.show()


# ARIMA модель
model_arima = ARIMA(data['Close'], order=(5,1,0))
fit_arima = model_arima.fit()

# ARMA модель
model_arma = SARIMAX(data['Close'], order=(5, 0, 0))
fit_arma = model_arma.fit()

# Ковзне середнє
rolling_mean = data['Close'].rolling(window=5).mean()

# Експоненційне згладжування
model_ses = SimpleExpSmoothing(data['Close']).fit()
model_hw = ExponentialSmoothing(data['Close']).fit()

# Прогнози
forecast_arima = fit_arima.forecast(steps=7,alpha=0.05)
forecast_arma = fit_arma.predict(start=len(data), end=len(data)+6)
forecast_rolling_mean = rolling_mean.iloc[-1:].repeat(7)
forecast_ses = model_ses.forecast(steps=7)
forecast_hw = model_hw.forecast(steps=7)

# Виведемо прогнози
print("Прогноз за допомогою ARIMA:", forecast_arima)
print("Прогноз за допомогою ARMA:", forecast_arma)
print("Прогноз за допомогою ковзного середнього:", forecast_rolling_mean)
print("Прогноз за допомогою експоненційного згладжування (просте):", forecast_ses)
print("Прогноз за допомогою експоненційного згладжування (гольцова-Вінтера):", forecast_hw)

# Порівняння
# Фактичні значення цін закриття
actual_values = data['Close'].iloc[-7:]
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Для ARIMA
mse_arima = mean_squared_error(actual_values, forecast_arima)
mae_arima = mean_absolute_error(actual_values, forecast_arima)

# Для ARMA
mse_arma = mean_squared_error(actual_values, forecast_arma)
mae_arma = mean_absolute_error(actual_values, forecast_arma)

# Для ковзного середнього
mse_rolling_mean = mean_squared_error(actual_values, forecast_rolling_mean)
mae_rolling_mean = mean_absolute_error(actual_values, forecast_rolling_mean)

# Для простого експоненційного згладжування
mse_ses = mean_squared_error(actual_values, forecast_ses)
mae_ses = mean_absolute_error(actual_values, forecast_ses)

# Для експоненційного згладжування з методом Гольцова-Вінтера
mse_hw = mean_squared_error(actual_values, forecast_hw)
mae_hw = mean_absolute_error(actual_values, forecast_hw)

# Виведемо значення MSE та MAE для порівняння
print("MSE для ARIMA:", mse_arima)
print("MAE для ARIMA:", mae_arima)
print("MSE для ARMA:", mse_arma)
print("MAE для ARMA:", mae_arma)
print("MSE для ковзного середнього:", mse_rolling_mean)
print("MAE для ковзного середнього:", mae_rolling_mean)
print("MSE для простого експоненційного згладжування:", mse_ses)
print("MAE для простого експоненційного згладжування:", mae_ses)
print("MSE для експоненційного згладжування (гольцова-Вінтера):", mse_hw)
print("MAE для експоненційного згладжування (гольцова-Вінтера):", mae_hw)