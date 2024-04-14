import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

# Загрузка набора данных
df = pd.read_csv('water_potability.csv')

# Приведение к числовому формату пустых ячеек
features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', "Potability"]
df[features] = df[features].fillna(df[features].mean())

# Выделение признаков и целевой переменной
X = df.drop('Potability', axis=1)  # Все признаки, кроме целевой переменной
y = df['Potability']  # Целевая переменная

# Масштабирование числовых признаков
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение данных на обучающий (0,9%) и тестовый (0,1%) наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# KMeans кластеризация
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
kmeans_labels = kmeans.labels_

# AgglomerativeClustering кластеризация
aggclust = AgglomerativeClustering(n_clusters=10)
aggclust.fit(X)
aggclust_labels = aggclust.labels_

# Оценка производительности моделей на основе коэффициента силуэта
print("Оценка производительности моделей на основе коэффициента силуэта")
print('KMeans Silhouette Score:', silhouette_score(X, kmeans_labels))
print('AgglomerativeClustering Silhouette Score:', silhouette_score(X, aggclust_labels))

# График кластеров для KMeans
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
plt.title('KMeans Clusters')
plt.show()

# График кластеров для AgglomerativeClustering
plt.scatter(X[:, 0], X[:, 1], c=aggclust_labels)
plt.title('AgglomerativeClustering Clusters')
plt.show()
