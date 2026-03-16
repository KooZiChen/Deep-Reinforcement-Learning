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
from huggingface_hub import login, HfApi
from huggingface_hub.repocard import metadata_eval_result, metadata_save

import imageio
from pathlib import Path
import datetime
import json
import tempfile
from tqdm import tqdm
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

env_id = "CartPole-v1"

env = gym.make(env_id, render_mode="rgb_array")

# env_eval = gym.make(env_id) # No longer needed

s_size = env.observation_space.shape[0]
# print(s_size)
a_size = env.action_space.n
# print(a_size)

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample())  # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample())  # Take a random action


cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": int(s_size),
    "action_space": int(a_size),
}


class Policy(
    nn.Module
):  # Policy is a subclass of nn.Module that encapsulates the network architecture and defines the computational graph via the forward function
    def __init__(self, s_size, a_size, h_size):
        super(
            Policy, self
        ).__init__()  # Find the parent class of Policy ( which is nn.Module) and run its initialization code on this specific object
        # Create two fully connected layers
        self.fc1 = nn.Linear(s_size, h_size)  # h_size stands for hidden layer
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = self.fc1(x)  # linear combination -> h_size
        x = F.relu(x)  # RELU activation function to remove the negative number
        logits = self.fc2(x)  # Second linear layer -> a_size
        return F.softmax(
            logits, dim=1
        )  # Softmax function to convert the logits to probabilities

    def act(self, state):
        state = (
            torch.from_numpy(state).float().unsqueeze(0)
        )  # Convert the state to a tensor and add a dimension to it
        probs = (
            self.forward(state).cpu()
        )  # Get the probabilities from the forward function and move them to the CPU
        m = Categorical(
            probs
        )  # Create a categorical distribution from the probabilities
        action = m.sample()  # Sample an action from the distribution
        return action.item(), m.log_prob(
            action
        )  # Return the action and its log probability


def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    tbar = tqdm(range(1, n_training_episodes + 1), desc="Training")
    for i_episode in tbar:
        saved_log_probs = []
        rewards = []
        state, _ = env.reset()  # Reset the environment
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)  # get the action
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)  # take an env step
            done = terminated or truncated
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]:  # reverse the rewards
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(
                rewards[t] + gamma * disc_return_t
            )  # calculate the return

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()

        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        # policy_loss = -objective function
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()  # clear the gradient
        policy_loss.backward()  # compute the gradient
        optimizer.step()  # update the policy

        if i_episode % print_every == 0:
            avg_score = np.mean(scores_deque)
            tbar.set_postfix(avg_score=f"{avg_score:.2f}")
            tbar.write(f"Episode {i_episode}\tAverage Score: {avg_score:.2f}")

    return scores


cartpole_policy = Policy(
    cartpole_hyperparameters["state_space"],
    cartpole_hyperparameters["action_space"],
    cartpole_hyperparameters["h_size"],
).to(device)

cartpole_optimizer = optim.Adam(
    cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"]
)

# Training or Loading the model
model_path = "unit_04/reinforce_cartpole.pt"
load_model = False

if os.path.exists(model_path):
    print(f"\nFound an existing model at {model_path}")
    choice = input("Do you want to load the existing model and skip training? (y/n): ").lower()
    if choice == 'y':
        load_model = True

if load_model:
    print(f"Loading model from {model_path}...")
    cartpole_policy = torch.load(model_path, weights_only=False)
    cartpole_policy.eval()
    scores = [] # No scores if we skip training
else:
    print("\n--- Starting Training ---")
    scores = reinforce(
        cartpole_policy,
        cartpole_optimizer,
        cartpole_hyperparameters["n_training_episodes"],
        cartpole_hyperparameters["max_t"],
        cartpole_hyperparameters["gamma"],
        100,
    )
    print("\nTraining complete! Saving model...")
    # Save the model
    torch.save(cartpole_policy, model_path)
    print(f"Model saved to {model_path}")

    print("\nGenerating training plot...")
    # Plot the training results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.title("REINFORCE CartPole-v1 Training Progress")
    plt.savefig("unit_04/reinforce_cartpole_rewards.png")
    print("Plot saved to unit_04/reinforce_cartpole_rewards.png")
    print("Displaying plot. CLOSE the plot window to continue...")
    plt.show()


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, _ = env.reset()
        step = 0
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


print("\nEvaluating agent...")
mean_reward, std_reward = evaluate_agent(
    env,
    cartpole_hyperparameters["max_t"],
    cartpole_hyperparameters["n_evaluation_episodes"],
    cartpole_policy,
)
print(f"Evaluation complete! Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")


def record_video(env, policy, out_directory, fps=30):
    images = []
    done = False
    state, _ = env.reset()
    img = env.render()
    images.append(img)
    while not done:
        action, _ = policy.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        img = env.render()
        images.append(img)
    imageio.mimsave(
        out_directory, [np.array(img) for i, img in enumerate(images)], fps=fps
    )


def push_to_hub(repo_id, model, hyperparameters, video_fps=30):

    _, repo_name = repo_id.split("/")
    api = HfApi()

    # Create evaluation environment with rgb_array mode for video recording
    eval_env = gym.make(hyperparameters["env_id"], render_mode="rgb_array")

    # Step 1: Create the repo
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        local_directory = Path(tmpdirname)

        # Step 2: Save the model
        torch.save(model, local_directory / "model.pt")

        # Step 3: Save the hyperparameters to JSON
        with open(local_directory / "hyperparameters.json", "w") as outfile:
            json.dump(hyperparameters, outfile)

        # Step 4: Evaluate the model and build JSON
        mean_reward, std_reward = evaluate_agent(
            eval_env,
            hyperparameters["max_t"],
            hyperparameters["n_evaluation_episodes"],
            model,
        )
        # Get datetime
        eval_datetime = datetime.datetime.now()
        eval_form_datetime = eval_datetime.isoformat()

        evaluate_data = {
            "env_id": hyperparameters["env_id"],
            "mean_reward": mean_reward,
            "n_evaluation_episodes": hyperparameters["n_evaluation_episodes"],
            "eval_datetime": eval_form_datetime,
        }

        # Write a JSON file
        with open(local_directory / "results.json", "w") as outfile:
            json.dump(evaluate_data, outfile)

        # Step 5: Create the model card
        env_name = hyperparameters["env_id"]

        metadata = {}
        metadata["tags"] = [
            env_name,
            "reinforce",
            "reinforcement-learning",
            "custom-implementation",
            "deep-rl-class",
        ]

        # Add metrics
        eval = metadata_eval_result(
            model_pretty_name=repo_name,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=env_name,
            dataset_id=env_name,
        )

        # Merges both dictionaries
        metadata = {**metadata, **eval}

        model_card = f"""
  # **Reinforce** Agent playing **{env_name}**
  This is a trained model of a **Reinforce** agent playing **{env_name}** .
  To learn to use this model and train yours check Unit 4 of the Deep Reinforcement Learning Course: https://huggingface.co/deep-rl-course/unit4/introduction
  """

        readme_path = local_directory / "README.md"
        readme = ""
        if readme_path.exists():
            with readme_path.open("r", encoding="utf8") as f:
                readme = f.read()
        else:
            readme = model_card

        with readme_path.open("w", encoding="utf-8") as f:
            f.write(readme)

        # Save our metrics to Readme metadata
        metadata_save(readme_path, metadata)

        # Step 6: Record a video
        video_path = local_directory / "replay.mp4"
        record_video(eval_env, model, video_path, video_fps)

        # Step 7. Push everything to the Hub
        api.upload_folder(
            repo_id=repo_id,
            folder_path=local_directory,
            path_in_repo=".",
        )

        print(f"Your model is pushed to the Hub. You can view your model here: {repo_url}")


login()

print("\nPushing model to Hugging Face Hub...")
api = HfApi()
try:
    user = api.whoami()
    username = user["name"]
    print(f"Authenticated as: {username}")
except Exception:
    print("Could not retrieve token information. Please ensure you are logged in with a 'Write' token.")
    username = "zichen" # Default fallback, though likely to fail again if wrong

repo_id = f"{username}/Reinforce-CartPole-v1"
print(f"Repository ID: {repo_id}")

push_to_hub(
    repo_id,
    cartpole_policy,
    cartpole_hyperparameters,
    video_fps=30,
)