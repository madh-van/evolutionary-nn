from sklearn.model_selection import train_test_split

import pandas as pd
import numpy

def handle_data(filepath="data/data.csv"):
    data = clear_up(filepath)
    data_min = data.min()
    data_max = data.max()

    # print (data_min)
    # print (data_max)

    train , cv = train_test_split(data, test_size = 0.15)

    x_train = numpy.array(train[['input_1', 'input_2']]).T
    y_train = numpy.array(train[['output_1', 'output_2']]).T

    x_cv = numpy.array(cv[['input_1', 'input_2']]).T
    y_cv = numpy.array(cv[['output_1', 'output_2']]).T

    return (x_train, y_train, x_cv, y_cv)

def clear_up(file_name="new_data.csv"):
    data = pd.read_csv(file_name, header=None)
    data.columns = ['input_1', 'input_2', 'output_1', 'output_2']
    data = data.drop_duplicates()
    data = (data - data.min()) / (data.max() - data.min())
    # data.plot(kind='density', subplots=True, layout=(2,2), sharex=False)
    # plt.show()
    return data

def find_average_loss(networks):
    t = 0

    for network in networks:
        t += network.loss

    avg = t / len(networks)
    return avg
