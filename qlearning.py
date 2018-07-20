from time import sleep
import numpy as np
import gym

env = gym.make('FrozenLake-v0')
inputCount = env.observation_space.n
actionsCount = env.action_space.n

Q = {}
for i in range(inputCount):
    Q[i] = np.random.rand(actionsCount)

lr = 0.33
lrDecay = 0.9999
gamma = 1.0
epsilon = 1.0
epsilonDecay = 0.97
episodes = 2000

for i in range(episodes):
    print("Episode {}/{}".format(i + 1, episodes))
    s = env.reset()
    done = False

    while not done:
        if np.random.random() < epsilon:
            a = np.random.randint(0, actionsCount)
        else:
            a = np.argmax(Q[s])

        newS, r, done, _ = env.step(a)
        Q[s][a] = Q[s][a] + lr * (r + gamma * np.max(Q[newS]) - Q[s][a])
        s = newS

        lr *= lrDecay

        if not r==0:
            epsilon *= epsilonDecay


print("")
print("Learning Rate :", lr)
print("Epsilon :", epsilon)

# Play game on 100 episodes
print("\nPlay Game on 100 episodes...")

avg_r = 0
for i in range(100):
    s = env.reset()
    done = False

    while not done:
        a = np.argmax(Q[s])
        newS, r, done, _ = env.step(a)
        s = newS

    avg_r += r/100.

print("Average reward on 100 episodes :", avg_r)
