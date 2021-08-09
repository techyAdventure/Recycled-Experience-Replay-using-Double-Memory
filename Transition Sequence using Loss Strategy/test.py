from agent import Agent
import gym
import random
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from environment import Environment
import time


env = gym.make('LunarLander-v2')
#env = gym.make('MountainCar-v0')

agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, seed=0)

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(10):
    state = env.reset()
    for j in range(200):
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        env.render()
       # time.sleep(0.4)
        if done:
            #time.sleep(1)
            break
