import numpy as np
import matplotlib.pyplot as plt

def pca(X, n_components):
    # Стандартизувати дані
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Щоб уникнути ділення на нуль
    X = (X - X_mean) / X_std

    # Обчислити коваріаційну матрицю
    cov_matrix = np.cov(X, rowvar=False)

    # Обчислити власні значення та власні вектори
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Відсортувати власні значення та власні вектори в порядку зменшення власних значень
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    # Відібрати перші n_components власних векторів
    components = eigenvectors[:, :n_components]

    # Перетворити дані на новий простір
    X_pca = np.dot(X, components)

    return X_pca, components, eigenvalues

# Приклад використання
X = np.random.rand(100, 4)  # Приклад згенерованих даних
n_components = 2
X_pca, components, eigenvalues = pca(X, n_components)

# Генерування експериментальних даних
data = np.random.rand(100, 5)  # Приклад згенерованих даних

# Відображення даних на екрані монітора у вигляді таблиці
print("Експериментальні дані:")
print(data)

# Стандартизація даних
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
data_std[data_std == 0] = 1
data_normalized = (data - data_mean) / data_std

# Побудова кореляційної матриці
correlation_matrix = np.corrcoef(data_normalized, rowvar=False)
print("\nКореляційна матриця:")
print(correlation_matrix)

# Перевірка значимості кореляційної матриці
# Наприклад, можна використовувати статистичний тест

# Розрахунок проекцій об'єктів на головні компоненти
X_pca, components, eigenvalues = pca(data, n_components=2)

# Побудова матриці даних, рахунків, помилок та навантажень
# Детальніше дивіться в реалізації PCA вище.

# Перевірка рівності сум вибіркових дисперсій
total_variance_original = np.sum(np.var(data, axis=0))
total_variance_pca = np.sum(eigenvalues)
print("Сума вибіркових дисперсій вихідних ознак:", total_variance_original)
print("Сума вибіркових дисперсій проекцій на головні компоненти:", total_variance_pca)

# Визначення відносної частки розкиду
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
print("\nВідносна частка розкиду:", explained_variance_ratio)

# Побудова матриці коваріації для проекцій об'єктів на головні компоненти
covariance_matrix_pca = np.cov(X_pca, rowvar=False)
print("\nМатриця коваріації для проекцій об'єктів на головні компоненти:")
print(covariance_matrix_pca)

# Побудова діаграми розсіювання
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Головна компонента 1')
plt.ylabel('Головна компонента 2')
plt.title('Діаграма розсіювання перших двох головних компонент')
plt.show()