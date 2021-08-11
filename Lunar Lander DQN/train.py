import torch
from agent import Agent
import gym
import random
import gym
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import time

env = gym.make('LunarLander-v2')
#env = gym.make('MountainCar-v0')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)


agent = Agent(state_size=8, action_size=4, seed=0)

# # # watch an untrained agent
# state = env.reset()
# #print(state, type(state))
# for j in range(200):

#     action = agent.act(state=state)
#     env.render()
#     state, reward, done, _ = env.step(action)
#     print(state)
#     if done:
#         break

# env.close()


def dqn(n_episodes=10_000, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.9995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    max_score = 100.0
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        # print(type(state))
        # quit()
        score = 0
        for t in range(max_t):
            # env.render()
            # time.sleep(1)
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            # print(state)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))

        if np.mean(scores_window) >= max_score:
            max_score = np.mean(scores_window)
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(),
                       'checkpoint.pth')
            # break
    return scores

start_time = time.time()
scores = dqn()
end_time = time.time()

scores_dqn_np = np.array(scores)
np.savetxt("scores_dqn_classic2.txt", scores_dqn_np)

def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "Execution time: %d hours : %02d minutes : %02d seconds" % (hour, minutes, seconds)

n = end_time-start_time
train_time = convert(n)
print(train_time)


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


scores_ma_dqn = moving_average(scores, n=100)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_ma_dqn)), scores_ma_dqn)
plt.ylabel('Scores')
plt.xlabel('Episodes')
plt.show()


# # load the weights from file
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

# for i in range(10):
#     state = env.reset()
#     for j in range(200):
#         action = agent.act(state)
#         env.render()
#         state, reward, done, _ = env.step(action)
#         if done:
#             break

# env.close()
