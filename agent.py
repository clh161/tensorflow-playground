from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from collections import deque
import random
import numpy as np


class Agent:

    def __init__(self, env, memory_size=1000, sample_size=32, gamma=0.99, model=None,
                 epsilon_decay=0.999,
                 epsilon_min=0.001, learning_rate=0.001):
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.epsilon = 1
        self.gamma = gamma
        self.sample_size = sample_size
        self.memory_size = memory_size
        self.env = env
        if model is None:
            model = self.build_model()
        self.model = model
        self.memory = deque()
        self.rewards = deque()
        self.epsilon_decay = epsilon_decay
        self.step = 0

    def build_model(self):
        model = Sequential()
        model.add(Dense(12, input_dim=4))
        model.add(Activation('relu'))
        model.add(Dense(12))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('linear'))
        optimizer = optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(np.array([state]))[0])

    def append_memory(self, state, state1, action, reward, done):
        self.memory.append([state, state1, action, reward, done])
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def replay(self):
        if self.memory_size > len(self.memory):
            return 0
        samples = random.sample(self.memory, self.sample_size)
        s_batch = [sample[0] for sample in samples]
        s1_batch = [sample[1] for sample in samples]
        a_batch = [sample[2] for sample in samples]
        r_batch = [sample[3] for sample in samples]
        done_batch = [sample[4] for sample in samples]
        predicts = self.model.predict(np.array(s_batch))
        q_values = self.model.predict(np.array(s1_batch))
        for i in range(0, len(predicts)):
            if done_batch[i]:
                predicts[i][a_batch[i]] = r_batch[i]
            else:
                predicts[i][a_batch[i]] = r_batch[i] + self.gamma * np.amax(q_values[i])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        loss = self.model.fit(x=np.array(s_batch), y=predicts, verbose=None).history['loss']
        self.model.save_weights('model-v1.h5')
        return np.average(loss)

    def train(self, episodes):
        for _ in range(episodes):
            state = self.env.reset()
            total_reward = 0
            for t in range(1000):
                self.step += 1
                action = self.get_action(state)
                state1, reward, done, _ = self.env.step(action)
                total_reward += reward
                self.append_memory(state, state1, action, reward, done)
                state = state1

                if done:
                    self.rewards.append(total_reward)
                    if len(self.rewards) > 10:
                        self.rewards.popleft()
                    loss = self.replay()
                    print("Steps: %d, Rewards: %.3f, Loss: %.3f Epsilon: %0.5f" % (
                        self.step, np.average(self.rewards), np.average(loss), self.epsilon))
                    break


