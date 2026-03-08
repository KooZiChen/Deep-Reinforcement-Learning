import gymnasium as gym
from stable_baselines3 import PPO
from huggingface_sb3 import package_to_hub
from huggingface_hub import login
from gymnasium.envs.registration import register

# 1. BYPASS: Force register v2 again for this session
try:
    register(
        id='LunarLander-v2',
        entry_point='gymnasium.envs.box2d:LunarLander',
        max_episode_steps=1000,
        reward_threshold=200,
    )
except:
    pass
# 1. Login to Hugging Face
login()

# 2. Define your parameters (must match your trained model)
env_id = "LunarLander-v2"
model_filename = "ppo-LunarLander-v2" # This looks for ppo-LunarLander-v2.zip
repo_id = "TheBestMoldyCheese/ppo-LunarLander-v2"

# 3. Load the saved model
# We don't need to define the env yet, SB3 loads the architecture from the zip
model = PPO.load(model_filename)

# 4. Create the evaluation environment
# This is required for package_to_hub to generate the replay video and metrics
eval_env = gym.make(env_id, render_mode="rgb_array")

# 5. Push to Hub
package_to_hub(
    model=model,
    model_name=model_filename,
    model_architecture="PPO",
    env_id=env_id,
    eval_env=eval_env,
    repo_id=repo_id,
    commit_message="Pushing pre-trained LunarLander model"
)