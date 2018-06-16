from __future__ import absolute_import, division, print_function

import numpy as np
import pandas
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt

# loaddataset
dataframe = pandas.read_csv("btc.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:1500, 0:9]
Y = dataset[:1500, 9]
test_X = dataset[1500:, 0:9]
test_Y = dataset[1500:, 9]
model = Sequential()
model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dense(100))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X, Y, epochs=1000, batch_size=128, verbose=0)
prediction = model.predict(x=np.array(test_X))
prediction = prediction.reshape((1, len(prediction)))[0]
# evaluation = model.evaluate(x=test_X, y=test_Y)
# plt.plot(test_Y, label='Actual')
# plt.plot(prediction, label='Prediction')
diff = np.divide(np.abs(np.subtract(test_Y, prediction)), test_Y / 100)
plt.plot(diff, label='diff')
# plt.plot(np.subtract(test_Y, prediction), label='Prediction')
plt.legend()
plt.show()
