import torch
import TD3
#import KRUCT
#import KR_UCT
import argparse
import copy
import gym
import re
import numpy as np
import glfw
import mujoco_py
#import roboschool
from scipy.stats import beta

def loadModel(policy,midstr):

    checkpoint = torch.load(midstr)
    policy.actor.load_state_dict(checkpoint['Actor'])
    policy.critic.load_state_dict(checkpoint['Critic'])
    return

def test(test_epoch):
    env = gym.make('FetchReach-v1')
    
    #state_dim = env.observation_space.shape[0]
    state_dim = env.observation_space["desired_goal"].shape[0] + env.observation_space["observation"].shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = TD3.TD3(state_dim,action_dim,max_action)
    #loadModel(policy,"models/model_36021.pt")
    filename = 'TD3_FetchReach-v1_543117_500000.0'
    policy.load(filename, './pytorch_models')

    for _ in range(1000):

        tr_obser = env.reset()
        total_reward = 0
        step_count = 0

        while(True):
            env.render()

            tr_obser = np.concatenate((tr_obser["desired_goal"],tr_obser["observation"]),axis = 0)
            tr_action = policy.select_action(tr_obser)
            tr_obser, tr_reward, is_terminal, _ = env.step(tr_action)

            total_reward += tr_reward
            step_count += 1

            if(is_terminal):
                
                break
    env.close()
    return

if(__name__ == "__main__"):

    test(100)

    