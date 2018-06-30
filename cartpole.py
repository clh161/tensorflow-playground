import gym

from agent import Agent

env = gym.make('CartPole-v0')

env.reset()
agent = Agent(env, [0, 1])
agent.train(stop_on_rewards=190)
