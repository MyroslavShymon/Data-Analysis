import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Дані таблиці
x_data = np.array([1.20, 1.57, 1.94, 2.31, 2.68, 3.05, 3.42, 3.79])
y_data = np.array([2.56, 2.06, 1.58, 1.25, 0.91, 0.66, 0.38, 0.21])

# Лінійна регресія (y = mx + b)
def linear_regression(x, m, b):
    return m * x + b

# Гіперболічна регресія (y = a / (x - b))
def hyperbolic_regression(x, a, b):
    return a / (x - b)

# Степенева регресія (y = ax^b)
def power_regression(x, a, b):
    return a * np.power(x, b)

# Показникова регресія (y = ab^x)
def exponential_regression(x, a, b):
    return a * np.power(b, x)

# Логарифмічна регресія (y = a * ln(x) + b)
def logarithmic_regression(x, a, b):
    return a * np.log(x) + b

# Функція для порівняння регресій
def compare_regression(x_data, y_data, func):
    popt, pcov = curve_fit(func, x_data, y_data)
    y_fit = func(x_data, *popt)
    residuals = y_data - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return popt, r_squared

# Визначення коефіцієнтів та оцінка якості регресії для кожного типу
linear_coeffs, linear_r_squared = compare_regression(x_data, y_data, linear_regression)
hyperbolic_coeffs, hyperbolic_r_squared = compare_regression(x_data, y_data, hyperbolic_regression)
power_coeffs, power_r_squared = compare_regression(x_data, y_data, power_regression)
exponential_coeffs, exponential_r_squared = compare_regression(x_data, y_data, exponential_regression)
logarithmic_coeffs, logarithmic_r_squared = compare_regression(x_data, y_data, logarithmic_regression)

# Друк результатів
print("Лінійна регресія: m =", linear_coeffs[0], "b =", linear_coeffs[1], "R^2 =", linear_r_squared)
print("Гіперболічна регресія: a =", hyperbolic_coeffs[0], "b =", hyperbolic_coeffs[1], "R^2 =", hyperbolic_r_squared)
print("Степенева регресія: a =", power_coeffs[0], "b =", power_coeffs[1], "R^2 =", power_r_squared)
print("Показникова регресія: a =", exponential_coeffs[0], "b =", exponential_coeffs[1], "R^2 =", exponential_r_squared)
print("Логарифмічна регресія: a =", logarithmic_coeffs[0], "b =", logarithmic_coeffs[1], "R^2 =", logarithmic_r_squared)

# Побудова графіків
plt.scatter(x_data, y_data, label='Табличні значення')
plt.plot(x_data, linear_regression(x_data, *linear_coeffs), label='Лінійна регресія')
plt.plot(x_data, hyperbolic_regression(x_data, *hyperbolic_coeffs), label='Гіперболічна регресія')
plt.plot(x_data, power_regression(x_data, *power_coeffs), label='Степенева регресія')
plt.plot(x_data, exponential_regression(x_data, *exponential_coeffs), label='Показникова регресія')
plt.plot(x_data, logarithmic_regression(x_data, *logarithmic_coeffs), label='Логарифмічна регресія')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()