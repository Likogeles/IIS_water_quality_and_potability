import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Загрузка данных
data = pd.read_csv("water_potability.csv")

features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

# Заполнение пропущенных значений (например, средними)
data[features] = data[features].fillna(data[features].mean())

# Разделение данных на обучающий и тестовый наборы
X = data[features]
y = data['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование данных (стандартизация)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение модели Лассо-регрессии
alpha = 0.1  # параметр регуляризации
lasso = Lasso(alpha=alpha)
lasso.fit(X_train_scaled, y_train)

# Предсказание на тестовом наборе
y_pred = lasso.predict(X_test_scaled)

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
accuracy = 1 - mse / y_test.var()

# Вывод точности модели
print(f"\nТочность модели: {accuracy * 100:.2f}%")
# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Печать трех лучших показателей
coef_dict = {}
for coef, feat in zip(lasso.coef_, features):
    coef_dict[feat] = coef

sorted_coef = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)

print("\nТри наиболее важных признака:")
for feature, coef in sorted_coef[:3]:
    print(f"{feature}: {coef}")
