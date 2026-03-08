import numpy as np

from collections import (
    deque,
)  # This is a double ended queue , it is used to store the last n rewards

import matplotlib.pyplot as plt

# gym
import gymnasium as gym


# PyTorch
import torch  # This is the base library that provides the Tensor , the fundamental data structure that looks like a multi-d array
import torch.nn as nn  # This is the neural network module
import torch.optim as optim  # This is the optimizer module , optimizer is used to
import torch.nn.functional as F  # This is the functional module , it provides the activation function like softmax and tanh
from torch.distributions import (
    Categorical,
)  # This handles action selection and exploration . It take those probabilities output by the softmax function and creates a mathematical object that can sample an action and calculate its log probability

# Hugging Face Hub
from huggingface_hub import login
import imageio

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

env_id = "CartPole-v1"

env = gym.make(env_id, render_mode="rgb_array")

env_eval = gym.make(env_id)

s_size = env.observation_space.shape[0]
# print(s_size)
a_size = env.action_space.n
# print(a_size)

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action

