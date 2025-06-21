from typing import Literal
import functools
import random
from collections import deque
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from flax.metrics.tensorboard import SummaryWriter
import optax

from src.tools import envs
from src.tools import utils


class DQNCNN(nnx.Module):
    def __init__(self, action_dim: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_features=4,
            out_features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            rngs=rngs,
        )
        self.conv3 = nnx.Conv(
            in_features=64,
            out_features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            rngs=rngs,
        )

        self.fc1 = nnx.Linear(in_features=3136, out_features=512, rngs=rngs)
        self.head = nnx.Linear(in_features=512, out_features=action_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = nnx.relu(self.fc1(x))
        qvalues = self.head(x)
        return qvalues


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_minibatch(self, batch_size: int = 32):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            "states": jnp.array(states),
            "actions": jnp.array(actions),
            "rewards": jnp.array(rewards),
            "next_states": jnp.array(next_states),
            "dones": jnp.array(dones),
        }

    def __len__(self):
        return len(self.buffer)


@nnx.jit
def train_step(online_network, target_network, data, optimizer, gamma: float = 0.99):

    def loss_fn(online_network):
        q_values = online_network(data["states"])
        q_values_selected = q_values[jnp.arange(len(data["actions"])), data["actions"]]

        next_q_values = target_network(data["next_states"])
        max_next_q_values = jnp.max(next_q_values, axis=1)
        targets = data["rewards"] + gamma * max_next_q_values * (1 - data["dones"])

        loss = jnp.mean((q_values_selected - targets) ** 2)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(online_network)
    optimizer.update(grads)
    return loss


def sync_target_network(online_network, target_network):
    """Copy weights from online network to target network."""
    state = nnx.state(online_network)
    nnx.update(target_network, state)


def main(env_id: str, outdir: str):

    utils.create_directory(outdir)

    env = envs.get_atari_env(env_id)
    action_dim: int = env.action_space.n

    online_network = DQNCNN(action_dim, rngs=nnx.Rngs(0))
    target_network = DQNCNN(action_dim, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(online_network, optax.adam(learning_rate=1e-4))
    replay_buffer = ReplayBuffer(maxlen=100_000)

    total_steps = 0
    while total_steps < 2_000_000:
        state, info = env.reset()
        ep_rewards, ep_steps = 0, 0

        while True:
            epsilon: float = max(0.1, 1.0 - total_steps / 1_000_000)  # Epsilon decay
            if epsilon > random.random():
                # Random action (exploration)
                action = env.action_space.sample()
            else:
                # Greedy action (exploitation)
                qvalues = online_networkn(state)
                import pdb; pdb.set_trace()  # fmt: skip

            next_state, reward, terminated, truncated, info = env.step(action)
            done: bool = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, done)

            # Update network
            if len(replay_buffer) > 1000 and n_steps % 4 == 0:
                batch_data = replay_buffer.sample_batch(32)
                loss = train_step(online_network, target_network, batch_data, optimizer)

            # Sync target network
            if n_steps % 10_000 == 0:
                sync_target_network(online_network, target_network)

            episode_reward += reward
            ep_steps += 1
            total_steps += 1

            if done:
                break

        with SummaryWriter(logdir=f"{outdir}/logs") as writer:
            writer.scalar("episode_reward", episode_reward, total_steps)
            writer.scalar("episode_steps", ep_steps, total_steps)


if __name__ == "__main__":
    main(env_id="Breakout-v4", outdir="out/dqn/breakout")
