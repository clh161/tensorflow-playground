import gym

import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import random
from collections import deque

env = gym.make('CartPole-v0')
model = Sequential()
model.add(Dense(4, input_dim=4))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('linear'))
optimizer = optimizers.SGD(lr=0.001)

model.compile(optimizer=optimizer, loss='mse')
# model.load_weights('model-v1.h5')

env.reset()
OBSERVE = 200000
MEMORY_SIZE = 100000
GAMMA = 0.99
BATCH_SIZE = 10000

t2 = 0
memory = deque()
rewards = deque()
while not None:
    s = env.reset()
    reward = 0
    for t in range(1000):
        t2 += 1
        if t2 < OBSERVE or random.uniform(0, 1) < 0.1:
            a = env.action_space.sample()
        else:
            p = model.predict(np.array([s]))
            a = np.argmax(p[0])

        s1, r, done, info = env.step(a)
        reward += r
        # env.render()
        memory.append([s, s1, a, r, done])
        s = s1
        if len(memory) > MEMORY_SIZE:
            memory.popleft()
        if t2 % BATCH_SIZE == 0 and len(memory) > BATCH_SIZE:
            samples = random.sample(memory, BATCH_SIZE)
            s_batch = [sample[0] for sample in samples]
            s1_batch = [sample[1] for sample in samples]
            a_batch = [sample[2] for sample in samples]
            r_batch = [sample[3] for sample in samples]
            done_batch = [sample[4] for sample in samples]
            predicts = model.predict(np.array(s_batch))
            q_values = model.predict(np.array(s1_batch))
            for i in range(0, len(predicts)):
                if done_batch[i]:
                    predicts[i][a_batch[i]] = r_batch[i]
                else:
                    predicts[i][a_batch[i]] = r_batch[i] + GAMMA * np.amax(q_values[i])
            loss = model.fit(x=np.array(s_batch), y=predicts, verbose=None).history['loss']
            # loss = model.train_on_batch(x=np.array(s_batch), y=predicts)
            print("Steps: %d, Rewards: %.3f, Loss: %.3f" % (t2, np.average(rewards), np.average(loss)))
            model.save_weights('model-v1.h5')


        if done:
            rewards.append(reward)
            if len(rewards) > 100:
                rewards.popleft()
            break
