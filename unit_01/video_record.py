import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
from huggingface_hub import HfApi, login
from gymnasium.envs.registration import register
import os
import shutil

# --- CONFIGURATION ---
env_id = "LunarLander-v2"
model_zip = "ppo-LunarLander-v2" # Looks for ppo-LunarLander-v2.zip
repo_id = "TheBestMoldyCheese/ppo-LunarLander-v2"
video_folder = "./temp_video"

# 1. FORCE REGISTER V2 (Bypass Deprecation)
try:
    register(
        id=env_id,
        entry_point='gymnasium.envs.box2d:LunarLander',
        max_episode_steps=1000,
        reward_threshold=200,
    )
except:
    pass

# 2. LOGIN TO HF
login() # Ensure your token has WRITE access
api = HfApi()

# 3. RECORD ONE SHORT VIDEO
print("Recording one episode for the preview...")
eval_env = gym.make(env_id, render_mode="rgb_array")
# Record only the first episode (index 0)
eval_env = RecordVideo(eval_env, video_folder=video_folder, name_prefix="rl-video", 
                       episode_trigger=lambda x: x == 0)

model = PPO.load(model_zip)
obs, info = eval_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    done = terminated or truncated
eval_env.close()

# 4. PREPARE AND UPLOAD
# SB3/HF Hub looks for 'replay.mp4' specifically
local_video = f"{video_folder}/rl-video-episode-0.mp4"
if os.path.exists(local_video):
    # Upload Video
    api.upload_file(
        path_or_fileobj=local_video,
        path_in_repo="replay.mp4",
        repo_id=repo_id,
        commit_message="Add manual video preview"
    )
    # Upload Model Zip
    api.upload_file(
        path_or_fileobj=f"{model_zip}.zip",
        path_in_repo=f"{model_zip}.zip",
        repo_id=repo_id,
        commit_message="Upload trained model weights"
    )
    print(f"Done! Check your repo: https://huggingface.co/{repo_id}")
else:
    print("Recording failed. Make sure 'ffmpeg' is installed on your system.")

# Cleanup local video folder
shutil.rmtree(video_folder, ignore_errors=True)