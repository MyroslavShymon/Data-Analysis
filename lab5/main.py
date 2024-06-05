import pandas as pd
import numpy as np

# Завантаження даних з CSV файлу
data = pd.read_csv("Sales.csv")

# Check for NaN values in 'Sales' and 'SKU' columns
if data['Sales'].isnull().values.any() or data['SKU'].isnull().values.any():
    print("Warning: 'Sales' column contains NaN values. Proceeding with NaN values.")

# Fill NaN values with a default value (e.g., 0)
data.fillna(0, inplace=True)

# Print the converted DataFrame
print(data)

import matplotlib.pyplot as plt

# Побудова графіка
plt.scatter(data['Date'], data['Sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales over Time')
plt.show()


# Обчислення відстаней
from scipy.spatial.distance import euclidean, cityblock
def standardized_euclidean(x, y):
    # Стандартизуємо вектори
    x_std = (x - np.mean(x)) / np.std(x)
    y_std = (y - np.mean(y)) / np.std(y)

    # Обчислюємо євклідову відстань між стандартизованими векторами
    distance = np.sqrt(np.sum((x_std - y_std)**2))

    return distance

# Обчислення відстаней
euclidean_distance = euclidean(data['Sales'], data['SKU'])
standardized_euclidean_distance = standardized_euclidean(data['Sales'], data['SKU'])
cityblock_distance = cityblock(data['Sales'], data['SKU'])

print("Euclidean Distance:", euclidean_distance)
print("Standardized Euclidean Distance:", standardized_euclidean_distance)
print("Cityblock Distance:", cityblock_distance)

from scipy.cluster.hierarchy import linkage, dendrogram
data_numeric = data.drop(columns=['Date', 'Product Name', 'City'])
# Зв'язування кластерів
single_linkage = linkage(data_numeric, method='single')
complete_linkage = linkage(data_numeric, method='complete')
average_linkage = linkage(data_numeric, method='average')

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cophenet

# Compute pairwise distances
pairwise_distances = pdist(data_numeric)

# Perform hierarchical clustering with different linkage methods
linkage_methods = ['single', 'complete', 'average']
cophenet_coefficients = {}

for i, method in enumerate(linkage_methods):
    linkage_matrix = linkage(data_numeric, method=method)
    cophenet_coefficients[method] = cophenet(linkage_matrix, pairwise_distances)[0]

# Print the table
print("Table 1: Cophenetic Correlation Coefficients")
print("Method\tk11\tk12\tk13")

for i, method1 in enumerate(linkage_methods):
    row = [method1]
    for j, method2 in enumerate(linkage_methods):
        if j < i:
            row.append(cophenet_coefficients[method2])
        elif j == i:
            row.append("-")
        else:
            row.append(0)  # Placeholder for remaining values
    print("\t".join(map(str, row)))

# Compute pairwise distances
pairwise_distances = pdist(data_numeric)

# Compute the cophenetic correlation coefficient for each linkage method
c_single = cophenet(single_linkage, pairwise_distances)[0]
c_complete = cophenet(complete_linkage, pairwise_distances)[0]
c_average = cophenet(average_linkage, pairwise_distances)[0]

print("Cophenetic Correlation Coefficient (Single Linkage):", c_single)
print("Cophenetic Correlation Coefficient (Complete Linkage):", c_complete)
print("Cophenetic Correlation Coefficient (Average Linkage):", c_average)
##################################################
# Визначення методу з найвищим значенням коефіцієнту
best_method = max([(c_single, 'single'), (c_complete, 'complete'), (c_average, 'average')])[1]

# Виведення найкращого коефіцієнту
print("Найкращий метод за коефіцієнтом кофенетичного кореляційного коефіцієнту:", best_method)

# Побудова дендрограми для найбільш ефективного методу
plt.figure(figsize=(10, 5))
plt.title(f'Hierarchical Clustering Dendrogram ({best_method.capitalize()} Linkage)')
dendrogram(locals()[f"{best_method}_linkage"])
plt.show()
##############

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Побудова моделі KMeans з різними кількостями кластерів
inertia_values = []
max_clusters = 10  # Максимальна кількість кластерів

for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data_numeric)
    inertia_values.append(kmeans.inertia_)

# Побудова графіка "ліктя"
plt.figure(figsize=(10, 5))
plt.plot(range(1, max_clusters + 1), inertia_values, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range(1, max_clusters + 1))
plt.grid(True)
plt.show()
#########################
import matplotlib.pyplot as plt

# Розрахунок центрів кластерів
cluster_centers = kmeans.cluster_centers_

# Відображення графічно знайдених кластерів та їх центрів
plt.figure(figsize=(10, 6))

# Відобразити кожен кластер окремим кольором
for cluster_index in range(kmeans.n_clusters):
    cluster_points = data.iloc[kmeans.labels_ == cluster_index]  # Використовуйте метод .iloc для отримання рядків за певним умовам
    plt.scatter(cluster_points['Sales'], cluster_points['SKU'], label=f'Cluster {cluster_index+1}')

# Відобразити центри кластерів
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='black', marker='x', label='Cluster Centers')

plt.title('Clusters and their Centers')
plt.xlabel('Sales')
plt.ylabel('SKU')
plt.legend()
plt.grid(True)
plt.show()