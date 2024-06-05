import numpy as np
import matplotlib.pyplot as plt

# Використаємо numpy.meshgrid для створення сітки точок
x = np.linspace(-500, 500, 100)
y = np.linspace(-500, 500, 100)
X, Y = np.meshgrid(x, y)

# Побудова поверхні за допомогою matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Z = 837.9657 - X * np.sin(np.sqrt(np.abs(X))) - Y * np.sin(np.sqrt(np.abs(Y)))
ax.plot_surface(X, Y, Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('F(x, y)')

plt.show()

def objective_function(x, y):
    return 837.9657 - x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))

def simple_stochastic_search(objective_function, bounds, n_iter=1000):
    best_solution = None
    best_value = float('inf')
    for _ in range(n_iter):
        x = np.random.uniform(bounds[0][0], bounds[0][1])
        y = np.random.uniform(bounds[1][0], bounds[1][1])
        value = objective_function(x, y)
        if value < best_value:
            best_solution = (x, y)
            best_value = value
    return best_solution, best_value

# Обмеження
bounds = [(-500, 500), (-500, 500)]

# Запускаємо алгоритм простого стохастичного пошуку
best_solution_sss, best_value_sss = simple_stochastic_search(objective_function, bounds)
print("Глобальний мінімум (Simple Stochastic Search):", best_solution_sss)
print("Мінімальне значення (Simple Stochastic Search):", best_value_sss)

def simulated_annealing(objective_function, bounds, n_iter=1000, initial_temp=100, cooling_rate=0.95):
    best_solution = None
    best_value = float('inf')
    current_solution = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
    current_value = objective_function(*current_solution)
    for i in range(n_iter):
        new_solution = [np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
        new_value = objective_function(*new_solution)
        if new_value < current_value:
            current_solution = new_solution
            current_value = new_value
            if new_value < best_value:
                best_solution = new_solution
                best_value = new_value
        else:
            p = np.exp(-(new_value - current_value) / initial_temp)
            if np.random.rand() < p:
                current_solution = new_solution
                current_value = new_value
        initial_temp *= cooling_rate
    return best_solution, best_value

# Запускаємо алгоритм методу імітації відпалу
best_solution_sa, best_value_sa = simulated_annealing(objective_function, bounds)
print("Глобальний мінімум (Simulated Annealing):", best_solution_sa)
print("Мінімальне значення (Simulated Annealing):", best_value_sa)