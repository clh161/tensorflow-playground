
import gym

from agent import Agent

env = gym.make('CartPole-v0')

env.reset()

agent = Agent(env)
agent.train(10000000)

