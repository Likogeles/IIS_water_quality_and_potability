import pandas as pd
import matplotlib.pyplot as plt


def data_hist():
    pd.set_option('display.max_columns', None)

    data = pd.read_csv('water_potability.csv')
    data.hist()
    plt.show()


if __name__ == '__main__':
    data_hist()
