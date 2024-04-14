import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка набора данных
df = pd.read_csv('water_potability.csv')

# Приведение к числовому формату пустых ячеек
features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', "Potability"]
df[features] = df[features].fillna(df[features].mean())

# Выделение признаков и целевой переменной
X = df.drop('Potability', axis=1)  # Все признаки, кроме целевой переменной
y = df['Potability']  # Целевая переменная

# Разделение данных на обучающий (0,9%) и тестовый (0,1%) наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Создание и обучение модели линейной регрессии
reg_lr = LinearRegression()
reg_lr.fit(X_train, y_train)

# Создание и обучение модели регрессии случайного леса
reg_rf = RandomForestRegressor(random_state=42)
reg_rf.fit(X_train, y_train)

# Прогноз по тестовым данным
y_pred_lr = reg_lr.predict(X_test)
y_pred_rf = reg_rf.predict(X_test)

# Оценка качества моделей
print("Среднеквадратичные ошибки моделей")
print('MSE LR:', mean_squared_error(y_test, y_pred_lr))
print('MSE RF:', mean_squared_error(y_test, y_pred_rf))
print()
print("Коэффициенты детерминации моделей")
print('R2 LR:', r2_score(y_test, y_pred_lr))
print('R2 RF:', r2_score(y_test, y_pred_rf))

# Визуализация результатов
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.regplot(x=y_test, y=y_pred_lr)
plt.title('Линейная регрессия')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.subplot(1, 2, 2)
sns.regplot(x=y_test, y=y_pred_rf)
plt.title('Регрессия случайного леса')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.tight_layout()
plt.show()
