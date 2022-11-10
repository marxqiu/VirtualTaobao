import argparse, math, os
import numpy as np
import gym
from gym import wrappers
import virtualTB

import torch
from torch.autograd import Variable
import torch.nn.utils as utils




env = gym.make('VirtualTB-v0')



env.seed(0)
torch.manual_seed(0)
np.random.seed(0)

agent = REINFORCE(128, env.observation_space.shape[0], env.action_space)

rewards = []
total_numsteps= 0 
updates = 0

for i_episode in range(10000):
    state = torch.Tensor([env.reset()])

    entropies = []
    log_probs = []
    epsisode_rewards = []
    while True:
        action, log_prob, entropy = agent.select_action(state)
        action = action.cpu()

        next_state, reward, done, _ = env.step(action.numpy()[0])

        entropies.append(entropy)
        log_probs.append(log_prob)
        epsisode_rewards.append(reward)
        state = torch.Tensor([next_state])

        if done:
            break

    agent.update_parameters(epsisode_rewards, log_probs, entropies, 0.99)

    rewards.append(np.sum(episode_reward))
    if i_episode % 10 == 0:
        episode_reward = 0
        episode_step = 0
        #evaluate the performance by 50 episodes
        for i in range(50):
            state = torch.Tensor([env.reset()])
            while True:
                action, _, _ = agent.select_action(state)
                action = action.cpu()

                next_state, reward, done, _ = env.step(action.numpy()[0])
                episode_reward += reward
                episode_step += 1

                next_state = torch.Tensor([next_state])

                state = next_state
                if done:
                    break

        # rewards.append(episode_reward)
        print("Episode: {}, total numsteps: {}, average reward: {}, CTR: {}".format(i_episode, episode_step, episode_reward / 50, episode_reward / episode_step / 10))

	
env.close()
