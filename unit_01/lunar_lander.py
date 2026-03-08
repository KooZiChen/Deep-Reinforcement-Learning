import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from huggingface_sb3 import package_to_hub
from huggingface_hub import login

# Force register v2 to bypass the DeprecatedEnv error
from gymnasium.envs.registration import register

try:
    register(
        id='LunarLander-v2',
        entry_point='gymnasium.envs.box2d:LunarLander',
        max_episode_steps=1000,
        reward_threshold=200,
    )
except:
    # If it's already registered, we just move on
    pass


login()


env_id = "LunarLander-v2"
env = make_vec_env(env_id, n_envs=16) 


model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1
)


model.learn(total_timesteps=1_000_000)


model.save("ppo-LunarLander-v2")
eval_env = gym.make(env_id, render_mode = "rgb_array")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


package_to_hub(
    model=model,
    model_name="ppo-LunarLander-v2",
    model_architecture="PPO",
    env_id=env_id,
    eval_env=eval_env,
    repo_id="TheBestMoldyCheese/ppo-LunarLander-v2",
    commit_message="Initial commit: Trained LunarLander-v2 with PPO"
)

