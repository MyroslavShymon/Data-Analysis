import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Завантаження даних з CSV
data = pd.read_csv('Sales2.csv')

# Відображення діаграми розсіювання
plt.figure(figsize=(10, 6))
plt.scatter(data['Sales'], data['Date'])
plt.xlabel('Sales')
plt.ylabel('Date')
plt.title('Scatter plot of Sales over Date')
plt.show()

# Оцінка кількості кластерів k
# На графіку розсіювання можна помітити відносно ясно визначені групи,
# так що можна спробувати використати метод "Ліктя" для вибору оптимальної кількості кластерів.
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[['Sales']])
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Використання методу "Ліктя" для визначення оптимальної кількості кластерів
# Зазвичай, "ліктьова точка" є місцем, де зміна відносної дії зменшується значно.
# У цьому випадку можна вибрати k=3.
optimal_k = 3

# Кластерний аналіз методом k-середніх
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['Sales']])

# Оцінка силуета
silhouette_avg = silhouette_score(data[['Sales']], data['Cluster'])
print(f"Average silhouette_score for {optimal_k} clusters:", silhouette_avg)

# Розрахунок центрів кластерів
cluster_centers = kmeans.cluster_centers_
print("Cluster centers:")
for i, center in enumerate(cluster_centers):
    print(f"Cluster {i+1}: {center}")

# Відображення графічного представлення кластерів
plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'red']
for i in range(optimal_k):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['Sales'], cluster_data['Date'], color=colors[i], label=f'Cluster {i+1}')

plt.scatter(cluster_centers, [data['Date'].min()] * optimal_k, color='black', marker='x', label='Cluster Centers')
plt.xlabel('Sales')
plt.ylabel('Date')
plt.title('Clustered Sales over Date')
plt.legend()
plt.show()