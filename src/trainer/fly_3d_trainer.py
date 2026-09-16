"""PyTorch Reinforcement Learning Trainer for 3D Fly Fruit Foraging.

Uses an Actor-Critic architecture for continuous flight control:
- State Space (12D): [left_odor, right_odor, center_odor, odor_gradient,
                     bearing_error, elevation_error, nearest_fruit_dist,
                     fly_vx, fly_vy, fly_vz, fly_altitude, fly_speed]
- Action Space (4D continuous): [thrust, pitch_rate, yaw_rate, roll_rate]

Includes asynchronous background training loop, model saving, and evaluation metrics.
"""
import os
import time
import math
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional, List
from src.world.world3d import World3D, World3DConfig


class FlyActorCritic(nn.Module):
    def __init__(self, state_dim: int = 12, action_dim: int = 4):
        super(FlyActorCritic, self).__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        # Actor head (mean of action distribution)
        self.actor_mean = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        # Actor std log
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head (state value V(s))
        self.critic = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        state_value = self.critic(features)
        return action_mean, action_std, state_value

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """Given a numpy state, returns (action, log_prob, state_value)."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action_mean, action_std, state_value = self.forward(state_t)

        if deterministic:
            action = action_mean.squeeze(0).detach().numpy()
            return action, 0.0, float(state_value.item())

        dist = torch.distributions.Normal(action_mean, action_std)
        action_sample = dist.sample()
        log_prob = dist.log_prob(action_sample).sum(dim=-1)

        action = action_sample.squeeze(0).detach().numpy()
        # Clip action to [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        return action, float(log_prob.item()), float(state_value.item())


def extract_state_vector(sensory: Dict[str, Any], world: World3D) -> np.ndarray:
    """Converts sensory dictionary and world state into a normalized 12D numpy state vector."""
    left_o = sensory["left_odor"]
    right_o = sensory["right_odor"]
    center_o = sensory["center_odor"]
    grad_o = sensory["odor_gradient"]
    bearing = sensory["bearing_error"] / math.pi
    elevation = sensory["elevation_error"] / math.pi
    dist = min(sensory["nearest_fruit_dist"], 30.0) / 30.0

    vx, vy, vz = world.fly.velocity / 5.0
    alt = min(sensory["fly_altitude"], 15.0) / 15.0
    speed = sensory["fly_speed"] / 5.0

    return np.array([
        left_o, right_o, center_o, grad_o,
        bearing, elevation, dist,
        vx, vy, vz, alt, speed
    ], dtype=np.float32)


class Fly3DTrainer:
    def __init__(self, world: Optional[World3D] = None, lr: float = 3e-4):
        self.world = world or World3D()
        self.model = FlyActorCritic(state_dim=12, action_dim=4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.load_model()

        self.is_training = False
        self.episode_count = 0
        self.total_steps = 0
        self.current_episode_reward = 0.0
        self.reward_history: List[float] = []
        self.fruits_history: List[int] = []

        self.best_reward = -float("inf")
        self.task: Optional[asyncio.Task] = None

    def train_step(self, max_ep_steps: int = 400) -> Dict[str, Any]:
        """Runs 1 complete training episode using Advantage Actor-Critic (A2C)."""
        self.world.reset()
        sensory = self.world.get_sensory_input()
        state = extract_state_vector(sensory, self.world)

        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        ep_reward = 0.0
        fruits_harvested = 0

        for step in range(max_ep_steps):
            action, log_prob, val = self.model.get_action(state, deterministic=False)
            sensory_next, reward, done, info = self.world.step(action)
            next_state = extract_state_vector(sensory_next, self.world)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(val)
            dones.append(done)

            ep_reward += reward
            self.total_steps += 1

            if info.get("harvested"):
                fruits_harvested += 1

            state = next_state
            if done:
                break

        # Compute discounted returns
        returns = []
        discounted_sum = 0.0
        gamma = 0.98
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                discounted_sum = 0.0
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions))
        returns_t = torch.FloatTensor(returns)

        action_means, action_stds, state_values = self.model(states_t)
        dist = torch.distributions.Normal(action_means, action_stds)
        log_probs_t = dist.log_prob(actions_t).sum(dim=-1)

        advantages = returns_t - state_values.squeeze(-1)
        actor_loss = -(log_probs_t * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()


        self.episode_count += 1
        self.reward_history.append(round(ep_reward, 2))
        self.fruits_history.append(fruits_harvested)
        if ep_reward > self.best_reward:
            self.best_reward = ep_reward
            self.save_model()

        return {
            "episode": self.episode_count,
            "reward": round(ep_reward, 2),
            "steps": len(states),
            "fruits_harvested": fruits_harvested,
            "loss": round(float(loss.item()), 4),
            "best_reward": round(self.best_reward, 2),
        }

    def save_model(self, filepath="fly3d_policy.pth"):
        try:
            torch.save(self.model.state_dict(), filepath)
        except Exception:
            pass

    def load_model(self, filepath="fly3d_policy.pth"):
        try:
            if os.path.exists(filepath):
                self.model.load_state_dict(torch.load(filepath))
                self.model.eval()
        except Exception:
            pass

    async def training_loop_async(self, ep_delay: float = 0.05):
        """Background async loop running training episodes."""
        self.is_training = True
        loop = asyncio.get_running_loop()
        try:
            while self.is_training:
                stats = await loop.run_in_executor(None, self.train_step)
                await asyncio.sleep(ep_delay)
        except asyncio.CancelledError:
            self.is_training = False

    def start_training(self):
        if not self.is_training:
            self.is_training = True
            loop = asyncio.get_event_loop()
            self.task = loop.create_task(self.training_loop_async())

    def pause_training(self):
        self.is_training = False
        if self.task and not self.task.done():
            self.task.cancel()

    def get_stats(self) -> Dict[str, Any]:
        recent_rewards = self.reward_history[-20:] if self.reward_history else [0.0]
        recent_fruits = self.fruits_history[-20:] if self.fruits_history else [0]
        return {
            "is_training": self.is_training,
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "mean_reward_last_20": round(float(np.mean(recent_rewards)), 2),
            "mean_fruits_last_20": round(float(np.mean(recent_fruits)), 2),
            "best_reward": round(self.best_reward if self.best_reward != -float("inf") else 0.0, 2),
            "reward_history": self.reward_history[-50:],
            "fruits_history": self.fruits_history[-50:],
        }
