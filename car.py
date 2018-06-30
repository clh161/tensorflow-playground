import gym
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from agent import Agent

env = gym.make('MountainCarContinuous-v0')

env.reset()
actions = [-1.0, -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1, 0.0, 1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1]

model = Sequential()
model.add(Dense(12, input_shape=env.observation_space.shape))
model.add(Activation('relu'))
model.add(Dense(12))
model.add(Activation('relu'))
model.add(Dense(len(actions)))
model.add(Activation('linear'))
optimizer = optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='mse')
agent = Agent(env, model=model,
              actions=actions)
agent.train(10000000)
