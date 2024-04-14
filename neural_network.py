import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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

# Создание и обучение модели
model = MLPClassifier(hidden_layer_sizes=(100, 50, 25, 10), max_iter=200)
model.fit(X_train, y_train)

# Прогноз по тестовым данным
y_pred = model.predict(X_test)

# Оценка качества модели
print("Оценки работы модели")
print('Accuracy:', accuracy_score(y_test, y_pred))

# График потерь (loss)
plt.plot(model.loss_curve_)
plt.title('График потерь')
plt.ylabel('Потери')
plt.xlabel('Эпохи')
plt.show()
