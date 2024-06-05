from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Метод для побудови дерева вирішальних правил
def treefit(X, y):
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    return clf

# Метод для обрізання дерева за рівнем або кількістю вузлів
def treeprune(t, criteria, value):
    if criteria == 'level':
        t.max_depth = value
    elif criteria == 'nodes':
        t.max_leaf_nodes = value

# Метод для відображення дерева
def treedisp(t):
    print(t.tree_)

# Метод для тестування дерева на тестовому наборі
def treetest(t, X_test, y_test):
    y_pred = t.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Метод для оцінки значень вихідного параметру Ycalc за допомогою дерева t та масиву незалежних змінних X
def treeval(t, X):
    Ycalc = t.predict(X)
    return Ycalc

# Завантаження набору даних з CSV файлу
def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Пункт 2: Завантаження набору даних
data_path = 'Sales.csv'
df = load_data_from_csv(data_path)

# Розділення набору даних на ознаки та цільову змінну
X = df[['Sales']]
y = df['Product Name']

# Розділення набору даних на тренувальний та тестовий
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Пункт 1: Побудова дерева вирішальних правил
clf = treefit(X_train, y_train)

# Пункт 3: Відображення дерева
plt.figure(figsize=(10,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_)
plt.show()

# Пункт 4: Тестування дерева на тестовому наборі
accuracy = treetest(clf, X_test, y_test)
print("Accuracy:", accuracy)

# Пункт 5: Розрахунок значень вихідного параметру Ycalc за допомогою дерева t
Ycalc = treeval(clf, X)
print("Predicted values:", Ycalc)

# Приклад тестового запису
test_sample = [[500]]  # Припустимо, ми маємо тестовий запис, де продажі дорівнюють 500

# Використання побудованого дерева для прийняття рішення на конкретному прикладі
predicted_class = clf.predict(test_sample)
print("Predicted class:", predicted_class)