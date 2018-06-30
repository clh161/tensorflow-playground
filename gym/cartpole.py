import gym

import numpy as np
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
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
optimizer = optimizers.SGD(lr=0.01)

model.compile(optimizer=optimizer, loss='mse')
# model.load_weights('model-v1.h5')

env.reset()
OBSERVE = 200000
MEMORY = 10000
GAMMA = 0.99

t2 = 0
D = deque()
store_s = deque()
store_s1 = deque()
store_a = deque()
store_r = deque()
store_d = deque()
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
        store_s.append(s)
        store_s1.append(s1)
        store_a.append(a)
        store_r.append(r)
        store_d.append(done)
        s = s1
        if len(store_s) > MEMORY:
            store_s.popleft()
            store_s1.popleft()
            store_a.popleft()
            store_r.popleft()
            store_d.popleft()
        if t2 % MEMORY == 0:
            ss = model.predict(np.array(store_s))
            ss1 = model.predict(np.array(store_s1))
            for i in range(0, len(ss)):
                if store_d[i]:
                    ss[i][store_a[i]] = store_r[i]
                else:
                    ss[i][store_a[i]] = store_r[i] + GAMMA * (np.amax(ss1[i]))
            loss = model.fit(x=np.array(store_s), y=ss, verbose=None, epochs=10, batch_size=32).history['loss']
            print("Steps: %d, Scores: %.3f, loss: %.3f" % (t2, np.average(rewards), np.average(loss)))
            model.save_weights('model-v1.h5')
        if done:
            rewards.append(reward)
            if len(rewards) > 1000:
                rewards.popleft()
            break
