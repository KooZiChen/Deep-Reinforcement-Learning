import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from huggingface_sb3 import package_to_hub
import ale_py
from huggingface_hub import login

# Register ALE environments in gymnasium (required in gymnasium >= 1.0)
gym.register_envs(ale_py)

# Log in to Hugging Face
login()

# 1. Set environment ID
env_id = "SpaceInvadersNoFrameskip-v4"

# 2. Create the Atari environment and wrap it with standard wrappers
# make_atari_env automatically handles frame skipping, grayscale, max-pooling, etc.
env = make_atari_env(env_id, n_envs=4, seed=0)

# 3. Stack 4 frames so the agent can see "motion"
env = VecFrameStack(env, n_stack=4)

# 4. Load the already trained DQN Model (Skipping initialization and training!)
model_name = "unit_03/dqn-SpaceInvadersNoFrameskip-v4"
model = DQN.load(model_name, env=env)

# 5. Total timesteps for commit message
total_timesteps = 1_000_000

# 6. Model is already saved locally from previous run

# 7. Evaluate the model
print("Evaluating model...")
eval_env = make_atari_env(env_id, n_envs=1, seed=0)
eval_env = VecFrameStack(eval_env, n_stack=4)

mean_reward, std_reward = evaluate_policy(
    model, eval_env, n_eval_episodes=10, deterministic=True
)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# 8. Push to Hugging Face Hub
print("Pushing to Hugging Face Hub...")
# For video rendering on HF hub, we need an RGB array render_mode
eval_env_hub = make_atari_env(
    env_id, n_envs=1, seed=0, env_kwargs={"render_mode": "rgb_array"}
)
eval_env_hub = VecFrameStack(eval_env_hub, n_stack=4)

package_to_hub(
    model=model,
    model_name="dqn-SpaceInvadersNoFrameskip-v4",
    model_architecture="DQN",
    env_id=env_id,
    eval_env=eval_env_hub,
    repo_id="TheBestMoldyCheese/dqn-SpaceInvadersNoFrameskip-v4",
    commit_message=f"Trained SpaceInvaders with DQN for {total_timesteps} steps",
)
