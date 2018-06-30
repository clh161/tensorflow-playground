import gym
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from agent import Agent

env = gym.make('Pendulum-v0')

env.reset()

actions = []
for i in range(-20, 20):
    actions.append(i / 10)
model = Sequential()
model.add(Dense(16, input_shape=env.observation_space.shape))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(len(actions)))
model.add(Activation('linear'))
optimizer = optimizers.SGD(lr=0.001)
model.compile(optimizer=optimizer, loss='mse')
agent = Agent(env, model=model,
              actions=actions)
agent.train(stop_on_rewards=90)
