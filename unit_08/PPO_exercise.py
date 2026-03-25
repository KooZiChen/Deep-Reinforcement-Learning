from dataclasses import dataclass

import random 
import numpy as np 

import gymnasium as gym 
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.distributions.categorical as Categorical


@dataclass
class Args:
    exp_name: str = "ppo-LunarLander-v3"
    env_id: str = "LunarLander-v3"
    seed: int = 1  # Starting number given to the Pseudorandom Number Generator
    torch_deterministic: bool = True  # If True, forces the GPU to use slower, reproducible math algorithms. This forces the GPU kernels to use a fixed math order to ensure bith level reproducibility
    cuda: bool = True  # If True, the CPU moves the neural network and rollout data to the VRAM for fast calculation

    # PPO
    # The Rollout Phase ( Collection Phase )
    total_timesteps: int = 500_000  # Total number of steps to train for (the training stops once the total steps is reached)
    num_steps: int = 128  # Each environment plays for 128 steps before the CPU pauses to send that data to the VRAM for an update
    num_envs: int = 4  # The number of parallel environments running at once

    # The Parameter
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True  # The agents starts learning fast and slowly cools down its learning speed as it gets closer to the end of the training
    gamma: float = 0.99  # Discount factor， the agent cares a lot about future reward ( landing safely ) rather than just immediate rewards ( not crashing right now)
    gae_lambda: float = 0.95  # Generalized Advantage Estimation， a method to reduce the variance of the advantage estimates
    update_epochs: int = 4  # Once the data is in the VRAM , how many times the GPU should re-read that same data to learn from it. Here , it looks at the data 4 times
    num_minibatches: int = 4  # Instead of calculating the gradient for all data at once , the GPU splits the rollout data into 4 smaller chunks to save VRAM and improve stability

    # Policy stability
    norm_adv: bool = True  # Normalizes the advantage ( How much better an action was than average) . This keeps the math stable
    clip_coef: float = 0.2  # The "Clipping" in PPO. It prevents the new policy from changing too drastically from the old one
    clip_vloss: bool = True  # If True , it clips the value loss function to prevent large updates to the critic
    ent_coef: float = 0.01  # Entropy coefficient. It encourages exploration by penalizing the agent for being too certain
    vf_coef: float = 0.5  # Value function coefficient. It weights the value loss in the total loss. Also means how much the value function (predicting score) matters compared to the policy ( choosing actions)
    max_grad_norm: float = (
        0.5  # Gradient clipping. It prevents the gradients from becoming too large
    )
    target_kl: float = None  # Target KL divergence. If set , it stops training if the KL divergence exceeds this value

    # derived (set in main)
    batch_size: int = 0  # Total number of steps collected per update (num_envs * num_steps) // the total amount of data the GPU sees in one learning cycle which is 4 x 128
    minibatch_size: int = 0  # Size of each minibatch (batch_size // num_minibatches) // the size of the chunks sent to the GPU cores 512/4 = 128

    """
TRAINING WORKFLOW DETAILED SEQUENCE:
Target: LunarLander-v3 | Hardware: CPU + GPU (CUDA)

PHASE 1: THE ROLLOUT (Data Collection) - Happens in System RAM
------------------------------------------------------------
1.  ENVIRONMENT STEP:
    - The CPU runs 'num_envs' (4) environments in parallel.
    - The Actor (Policy) predicts an action for the current state.
    - The Critic predicts the 'Value' (expected reward) for that state.
2.  BUFFER STORAGE:
    - Observations, Actions, Log_probs, Rewards, and Values are stored.
    - This repeats for 'num_steps' (128) per environment.
3.  SYNC POINT:
    - Once 128 steps * 4 envs = 512 total steps are collected, the simulation pauses.
    - Advantage (A_t) and Returns (G_t) are calculated using GAE (Generalized Advantage Estimation).
    - The return is needed by the critic , while the advantage is needed by the actor

PHASE 2: THE DATA HANDOVER (The PCIe Bridge)
------------------------------------------------------------
4.  VRAM TRANSFER:
    - The block of 512 transitions is converted to PyTorch Tensors.
    - These tensors are moved from System RAM to GPU VRAM (cuda:0).
    - batch_size = 512.

PHASE 3: THE OPTIMIZATION (The Learning) - Happens in GPU Cores
------------------------------------------------------------
5.  SHUFFLE & MINIBATCH:
    - The 512 steps are shuffled to break temporal correlation.
    - Data is split into 'num_minibatches' (4).
    - minibatch_size = 512 / 4 = 128 steps per GPU Kernel launch.
6.  EPOCH LOOP (Repeats 'update_epochs' = 4 times):
    a. ACTOR UPDATE: 
       - Calculate New_Log_Probs.
       - Calculate Ratio (New / Old).
       - Apply CLIP_COEF (0.2) to the Ratio * Advantage. -> Minimize -L_clip.
    b. CRITIC UPDATE: 
       - Calculate MSE between predicted Value and Actual Return.
       - Apply CLIP_VLOSS (Safety). -> Minimize VF_COEF * L_vf.
    c. ENTROPY BONUS: 
       - Calculate Action Diversity. -> Maximize S (Curiosity).
7.  BACKPROPAGATION:
    - GPU calculates the Total Loss gradient.
    - 'max_grad_norm' (0.5) clips the gradients to prevent "Explosion."
    - Optimizer (Adam) updates the Neural Network weights.

PHASE 4: THE REPEAT
------------------------------------------------------------
8.  CLEANUP: VRAM is cleared of the old 512-step batch.
9.  RESUME: CPU starts the next 128 steps using the NEWLY UPDATED weights.
    - This cycle repeats until 'total_timesteps' (500,000) is reached.
"""


def layer_init(layer , std = np.sqrt(2) , bias_const = 0.0): # the std scales the eigenvalues from 1 to std . Thus,  var = std^2 . For ReLu/tanh hidden layers : std = sqrt(2) compensates for the ~50% variance reduction (signal attenuation) caused by the activation's non-linearity, maintaining a stable forward-pass of variance of 1.0
    nn.init.orthogonal_(layer.weight , std)  # Orthogonal initialization is a method of initializing the weights of a neural network such that the weight matrices are orthogonal. This helps to prevent the vanishing or exploding gradient problem during training.
    nn.init.constant_(layer.bias , bias_const)  # Constant initialization is a method of initializing the weights of a neural network such that the weights are all set to the same constant value. This is often used for the bias terms of the neural network.
    return layer

class Agent(nn.Module):
    def __init__ (self , envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod() # prod() is used to calculate the product of all the elements in the array. In this case, it is used to calculate the product of all the elements in the observation space of the environment.
        act_dim = envs.single_action_space.n # n is used to get the number of actions in the environment.

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim , 64)), # The first layer takes the observation as input and outputs 64 values.
            nn.Tanh(), # The Tanh activation function squashes the output of the first layer to a range of -1 to 1.
            layer_init(nn.Linear(64 , 64)), # The second layer takes the output of the first layer and outputs 64 values.
            nn.Tanh(), # The Tanh activation function squashes the output of the second layer to a range of -1 to 1.
            layer_init(nn.Linear(64 , 1) , std = 1.0), # The third layer takes the output of the second layer and outputs 1 value.
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim , 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64 , 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64 , act_dim) , std = 0.01),
        )

    def get_value(self , x):
        return self.critic(x)

    def get_action_and_value(self , x , action = None):
        logits = self.actor(x)
        dist = Categorical(logits = logits)
        if action is None:
            action = dist.sample()
        return action , dist.log_prob(action) , dist.entropy() , self.critic(x)

def make_env(env_id , index , )

def train(args : Args) : 
    args.batch_size = args.num_steps * args.num_envs
    args.minibatch_size = args.batch_size // args.num_minibatches
    
    random.seed(args.seed)
    np.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i) for i in range(args.num_envs)]
    )




if __name__ == "__main__":
    args = Args()
    train(args)