import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from my_regression import scaler


def Nan_to_zero(a):
    if str(a) == "nan":
        return 0
    return a


def my_regression(data):
    # Подготовка данных
    x = data.drop('Potability', axis=1)  # Все признаки, кроме целевой переменной
    y = data['Potability']  # Целевая переменная

    # Разделение на обучающий и тестовый наборы данных
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Масштабирование целевой переменной
    y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    # Обучение решающего дерева
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(x_train, y_train_scaled)

    # Предсказание на тестовом наборе
    y_predict_tree = tree_reg.predict(x_test)

    # Рассчет метрик
    mid_square_tree = np.round(np.sqrt(metrics.mean_squared_error(y_test_scaled, y_predict_tree)), 3)
    coeff_determ_tree = np.round(metrics.r2_score(y_test_scaled, y_predict_tree), 2)

    # Визуализация результатов
    plt.plot(y_test_scaled, c="red", label="y тестовые")
    plt.plot(y_predict_tree, c="green", label=f"y предсказанные\nСр^2 = {mid_square_tree}")
    plt.legend(loc='upper right')
    plt.title("Регрессия с решающим деревом")
    plt.show()


def classification(data):
    # Определение групп (Low, Medium, High)
    bins = [0.0, 6.0, 8.0, float('inf')]
    group_labels = ['Acid_water', 'Slightly_alkaline_water', 'Alkaline_water']

    # Разделение данных на признаки (X) и целевую переменную (y)
    X = data[['ph']]  # pH как признак
    y = data['Potability']  # Признак, который мы пытаемся предсказать

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание модели дерева решений
    model = DecisionTreeClassifier()

    # Обучение модели на обучающем наборе данных
    model.fit(X_train, y_train)

    # Предсказание на тестовом наборе данных
    y_pred = model.predict(X_test)

    # Оценка точности модели
    accuracy = accuracy_score(y_test, y_pred)
    print("Точность модели: {:.2f}%".format(accuracy * 100))


def data_hist():
    pd.set_option('display.max_columns', None)

    data = pd.read_csv('water_potability.csv')

    # Приведение к числовому формату пустых ячеек
    # data['ph'] = data['ph'].apply(Nan_to_zero)
    # data['Hardness'] = data['Hardness'].apply(Nan_to_zero)
    # data['Solids'] = data['Solids'].apply(Nan_to_zero)
    # data['Chloramines'] = data['Chloramines'].apply(Nan_to_zero)
    # data['Sulfate'] = data['Sulfate'].apply(Nan_to_zero)
    # data['Conductivity'] = data['Conductivity'].apply(Nan_to_zero)
    # data['Organic_carbon'] = data['Organic_carbon'].apply(Nan_to_zero)
    # data['Trihalomethanes'] = data['Trihalomethanes'].apply(Nan_to_zero)
    # data['Turbidity'] = data['Turbidity'].apply(Nan_to_zero)

    # Приведение к числовому формату пустых ячеек
    data['ph'] = data['ph'].fillna(data['ph'].mean())
    data['Hardness'] = data['Hardness'].fillna(data['Hardness'].mean())
    data['Solids'] = data['Solids'].fillna(data['Solids'].mean())
    data['Chloramines'] = data['Chloramines'].fillna(data['Chloramines'].mean())
    data['Sulfate'] = data['Sulfate'].fillna(data['Sulfate'].mean())
    data['Conductivity'] = data['Conductivity'].fillna(data['Conductivity'].mean())
    data['Organic_carbon'] = data['Organic_carbon'].fillna(data['Organic_carbon'].mean())
    data['Trihalomethanes'] = data['Trihalomethanes'].fillna(data['Trihalomethanes'].mean())
    data['Turbidity'] = data['Turbidity'].fillna(data['Turbidity'].mean())

    # Вывод данных
    print("Данные:")
    print(data)
    print()

    # Вывод статистики
    print("Статистика:")
    print(data.describe())
    print()

    my_regression(data)

    # classification(data)

    # Гистограммы:
    # data.hist()
    # plt.show()


if __name__ == '__main__':
    data_hist()
