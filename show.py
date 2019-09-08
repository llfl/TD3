import numpy as np
import torch
import gym
import argparse
import os
#import mujoco
import utils
import TD3
import OurDDPG
import DDPG

def moduleShow(args):
    env = gym.make(args.env_name)
    state_dim = env.observation_space["observation"].shape[0] + env.observation_space["desired_goal"].shape[0]
    #state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    # Initialize policy
    if args.policy_name == "TD3": policy = TD3.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "OurDDPG": policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG": policy = DDPG.DDPG(state_dim, action_dim, max_action)

    replay_buffer = utils.ReplayBuffer()
    
    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy)] 
    obs = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")                    # Policy name
    parser.add_argument("--env_name", default="FetchReach-v1")
    parser.add_argument("--module_name", default="null")
    args = parser.parse_args()
    moduleShow(args)