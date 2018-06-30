from __future__ import absolute_import, division, print_function

import numpy as np
import pandas
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt

train_size = 1800
targeted_coin = 'btc'
coins = ['btc', 'eos', 'eth', 'xrp', 'ltc', 'dash', 'mco', 'bnb']
track_size = 10
prediction_interval = 3
model_name = "{}-{}-{}-{}.h5".format(targeted_coin, '-'.join(sorted(coins)), track_size, prediction_interval)
model_path = 'crypto/models/{}'.format(model_name)


def get_data():
    x = None
    y = None
    test_x = None
    test_y = None
    for i in range(len(coins)):
        coin = coins[i]
        data_set = pandas.read_csv("crypto/data/{}.csv".format(coin), header=None)
        values = data_set.values
        d = []
        for j in range(len(values) - track_size - prediction_interval):
            d.append(values[j:j + track_size, 1:].reshape((1, track_size))[0])
        if x is None:
            x = d[:train_size]
            test_x = d[train_size:]
        else:
            x = np.concatenate((x, d[:train_size]), axis=1)
            test_x = np.concatenate((test_x, d[train_size:]), axis=1)
        if coin == targeted_coin:
            y = values[track_size + prediction_interval:train_size + track_size + prediction_interval, 1:]
            test_y = values[train_size + track_size + prediction_interval:len(values), 1:]
    return x, y, test_x, test_y


def build_model(input_size):
    model = Sequential()
    model.add(Dense(input_size, input_dim=input_size, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1000))
    model.add(Dense(1000))
    model.add(Dense(1, kernel_initializer='normal'))
    optimizer = optimizers.Adam(lr=0.00000005)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    try:
        model.load_weights(model_path)
    except:
        print("Cannot load model")
    return model


def save_model_weights(model):
    model.save_weights(model_path)


def train(model, x, y, training_steps=100, epochs=100, batch_size=128):
    for i in range(training_steps):
        history = model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
        save_model_weights(model)
        print("{:d}/{:d} Loss: {:f}".format(i + 1, training_steps, np.average(history.history.get('loss'))))


def test(model, test_x, test_y):
    actual = test_y.reshape((1, len(test_y)))[0]
    prediction = model.predict(x=np.array(test_x))
    prediction = prediction.reshape((1, len(prediction)))[0]
    diff = np.divide(np.abs(np.subtract(actual, prediction)), actual / 100)
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Price')
    ax1.plot(test_y, label='Actual Price')
    ax1.plot(prediction, label='Predicted Price')
    ax1.set_ylim([prediction.min() * 0.8, prediction.max() * 1.2])
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.set_ylabel('Diff (%)')
    ax2.plot(diff, color='red', label='Diff (%)')
    ax2.set_ylim([0, 25])
    ax2.legend(loc=2)
    plt.show()
    print("Average of diff: {:.2f}%".format(np.average(diff)))


def run():
    x, y, test_x, test_y = get_data()
    print(x)
    model = build_model(len(x[0]))
    train(model, x, y)
    test(model, test_x, test_y)


run()
