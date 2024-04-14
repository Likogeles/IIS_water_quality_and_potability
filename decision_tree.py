import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Загрузка набора данных
df = pd.read_csv('water_potability.csv')

# Приведение к числовому формату пустых ячеек
features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']
df[features] = df[features].fillna(df[features].mean())

# Выделение признаков и целевой переменной
X = df.drop('Potability', axis=1)  # Все признаки, кроме целевой переменной
y = df['Potability']  # Целевая переменная

# Разделение данных на обучающий (0,9%) и тестовый (0,1%) наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Создание и обучение модели дерева решений
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)

# Создание и обучение модели случайного леса
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Прогноз по тестовым данным
y_pred_dt = model_dt.predict(X_test)
y_pred_rf = model_rf.predict(X_test)

# Оценка качества моделей
accuracy_dt = accuracy_score(y_test, y_pred_dt)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Оценки работы моделей")
print('Accuracy DecisionTreeClassifier: ', accuracy_dt)
print('Accuracy RandomForestClassifier: ', accuracy_rf)

# Сравнение важности характеристик моделей
print("Оценки важности признаков")
df_features = pd.DataFrame({'Признак': X.columns, 'DT': model_dt.feature_importances_, 'RF': model_rf.feature_importances_,
                            'Среднее': (model_dt.feature_importances_ + model_rf.feature_importances_) / 2})
print(df_features)

# График важности характеристик по лучшей модели
plt.figure(figsize=(10,8))
if accuracy_dt > accuracy_rf:
    sns.barplot(x=df_features['DT'], y=df_features['Признак'])
else:
    sns.barplot(x=df_features['RF'], y=df_features['Признак'])
plt.title('Ранжирование признаков')
plt.xlabel('Значительность')
plt.ylabel('Признак')
plt.show()
