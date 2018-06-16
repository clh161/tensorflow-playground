from __future__ import absolute_import, division, print_function

import numpy as np
import pandas
from keras.layers import Dense
from keras import optimizers
from keras.models import Sequential
from matplotlib import pyplot as plt

tran_size = 1500
targeted_coin = 'btc'
coins = ['btc', 'eos', 'eth', 'xrp', 'ltc', 'dash']
X = None
for i in range(len(coins)):
    coin = coins[i]
    dataframe = pandas.read_csv(coin + ".csv", header=None)
    values = dataframe.values
    if X is None:
        X = values[:tran_size, 0:-1]
        test_X = values[tran_size:, 0:-1]
    else:
        X = np.concatenate((X, values[:tran_size, 0:-1]), axis=1)
        test_X = np.concatenate((test_X, values[tran_size:, 0:-1]), axis=1)
    if coin == targeted_coin:
        Y = values[:tran_size, -1:]
        test_Y = values[tran_size:, -1:]

model = Sequential()
model.add(Dense(len(X[0]), input_dim=len(X[0]), kernel_initializer='normal', activation='relu'))
model.add(Dense(1000))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
optimizer = optimizers.Adam(lr=0.00001)

model.compile(loss='mean_squared_error', optimizer=optimizer)
history = model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

actual = test_Y.reshape((1, len(test_Y)))[0]
prediction = model.predict(x=np.array(test_X))
prediction = prediction.reshape((1, len(prediction)))[0]
diff = np.divide(np.abs(np.subtract(actual, prediction)), actual / 100)
fig, ax1 = plt.subplots()
ax1.set_ylabel('Price')
ax1.plot(test_Y, label='Actual Price')
ax1.plot(prediction, label='Predicted Price')
ax1.set_ylim([prediction.min() * 0.8, prediction.max() * 1.2])
ax1.legend()

ax2 = ax1.twinx()
ax2.set_ylabel('Diff (%)')
ax2.plot(diff, color='red', label='Diff (%)')
ax2.set_ylim([0, 25])
ax2.legend(loc=2)

evaluation = model.evaluate(x=test_X, y=test_Y)
plt.show()

print("Average of diff: {:.2f}%".format(np.average(diff)))
