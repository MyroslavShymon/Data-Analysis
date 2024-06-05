import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import csv
import random
import os

# Шлях до файлу
file_path = 'Sales.csv'

# Перевіряємо, чи файл вже існує
if not os.path.exists(file_path):
    # Кількість записів
    num_records = int(input("Введіть кількість записів: "))

    # Кількість продуктів
    num_products = 10

    # Назви продуктів
    products = [f"Product_{i}" for i in range(1, num_products + 1)]

    # Відкриваємо CSV-файл для запису
    with open(file_path, 'w', newline='') as csvfile:
        # Створюємо об'єкт для запису в CSV-файл
        writer = csv.writer(csvfile)

        # Записуємо назви колонок
        writer.writerow(products)

        # Генеруємо випадкові записи
        for _ in range(num_records):
            # Генеруємо рандомну послідовність чисел 0 або 1 для кожного продукту
            record = [random.randint(0, 1) for _ in range(num_products)]

            # Записуємо запис у CSV-файл
            writer.writerow(record)

    print("CSV-файл згенеровано успішно!")
else:
    print("Файл уже існує, генерація не потрібна.")

data = pd.read_csv("Sales.csv")
print(data)

# Застосовуємо алгоритм Apriori
frequent_itemsets = apriori(data, min_support=0.01, use_colnames=True)

# Знаходимо асоціативні правила
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Виводимо асоціативні правила
print(rules)