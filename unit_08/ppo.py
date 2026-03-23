"""
Unit 8 - PPO from scratch with CleanRL-style single-file implementation.
Environment: LunarLander-v2
Reference: https://huggingface.co/learn/deep-rl-course/unit8/hands-on-cleanrl
"""

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


# ── Hyperparameters ────────────────────────────────────────────────────────────

@dataclass
class Args:
    exp_name: str = "ppo-LunarLander-v3"
    env_id: str = "LunarLander-v3"
    seed: int = 1
    torch_deterministic: bool = True    
    cuda: bool = True

    # PPO
    total_timesteps: int = 500_000 #Total number of steps to train for 
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 128          # steps per rollout per env
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    # derived (set in main)
    batch_size: int = 0
    minibatch_size: int = 0


# ── Network ────────────────────────────────────────────────────────────────────

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = envs.single_action_space.n

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


# ── Training ───────────────────────────────────────────────────────────────────

def make_env(env_id, seed, idx, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk


def train(args: Args):
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" +
                    "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]))

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # envs
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, i, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # rollout storage
    obs      = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions  = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards  = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones    = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values   = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs  = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ── collect rollout ────────────────────────────────────────────────────
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step]  = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step]  = action
            logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device)
            next_obs  = torch.tensor(next_obs_np, dtype=torch.float32).to(device)
            next_done = torch.tensor(done, dtype=torch.float32).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']:.2f}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # ── GAE ────────────────────────────────────────────────────────────────
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten batch
        b_obs      = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions  = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns  = returns.reshape(-1)
        b_values   = values.reshape(-1)

        # ── PPO update ─────────────────────────────────────────────────────────
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate",       optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss",          v_loss.item(),                   global_step)
        writer.add_scalar("losses/policy_loss",         pg_loss.item(),                  global_step)
        writer.add_scalar("losses/entropy",             entropy_loss.item(),             global_step)
        writer.add_scalar("losses/approx_kl",           approx_kl.item(),                global_step)
        writer.add_scalar("losses/clipfrac",            np.mean(clipfracs),              global_step)
        writer.add_scalar("losses/explained_variance",  explained_var,                   global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)),  global_step)

        if update % 10 == 0:
            print(f"Update {update}/{num_updates} | SPS: {int(global_step / (time.time() - start_time))}")

    envs.close()
    writer.close()
    return agent, args


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = Args()
    agent, args = train(args)

    # save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{args.exp_name}.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"Model saved to {model_path}")
