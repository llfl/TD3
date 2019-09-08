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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")                    # Policy name
    parser.add_argument("--env_name", default="FetchReach-v1")
    parser.add_argument("--module_name", default="null")
    args = parser.parse_args()
    moduleShow(args)