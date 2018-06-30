from collections import deque

import gym

from agent import Agent

env = gym.make('Pendulum-v0')
# model.load_weights('model-v1.h5')

env.reset()

agent = Agent(env)
agent.train(10000000)

