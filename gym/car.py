import gym

import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import random
from copy import deepcopy

env = gym.make('MountainCarContinuous-v0')
model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('linear'))
optimizer = optimizers.SGD(lr=0.001)

model.compile(optimizer=optimizer, loss='mse')
# model.load_weights('model-v1.h5')

# env = gym.make('Pendulum-v0')
env.reset()
success = 0
while success < 2000:
    observation = env.reset()
    observation = [observation]
    observations = []
    actions = []
    reward = 0
    for t in range(1000):
        # env.render()
        if random.uniform(0, 1) < 1 or t == 0:
            action = env.action_space.sample()
        else:
            action = model.predict(np.array([observation]))
        o, r, done, info = env.step(action)
        observation = o.reshape(1, 2)[0]
        reward += r
        actions.append(action)
        observations.append(observation)
        if done:
            break
    if reward > 50:
        success += 1
        history = model.fit(x=np.array(observations), y=np.array(actions), verbose=0, epochs=100, batch_size=16)
        losses = history.history.get('loss')
        print("{:d}:{:f} Loss: {:f}".format(success, reward, np.average(losses)))
        model.save_weights('model-v1.h5')
