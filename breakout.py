import gym
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D

from agent import Agent

env = gym.make('Breakout-v0')

state = env.reset()
action_space = env.action_space

actions = [0, 1, 2, 3]
model = Sequential()
model.add(Conv2D(input_shape=env.observation_space.shape, kernel_size=4, filters=1))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(len(actions)))
model.add(Activation('linear'))
optimizer = optimizers.SGD(lr=0.000000001)
model.compile(optimizer=optimizer, loss='mse')
agent = Agent(env, model=model,
              actions=actions)
agent.train(stop_on_rewards=90)
