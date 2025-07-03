import random
from collections import deque
from pathlib import Path
import shutil
import pickle

import lz4.frame as lz4f
import jax.numpy as jnp
import numpy as np
from flax import nnx
import wandb

import optax

from src import utils


class DQNCNN(nnx.Module):
    def __init__(self, action_dim: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_features=4,
            out_features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            rngs=rngs,
        )
        self.norm1 = nnx.GroupNorm(num_features=32, num_groups=8, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=32,
            out_features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            rngs=rngs,
        )
        self.norm2 = nnx.GroupNorm(num_features=64, num_groups=8, rngs=rngs)
        self.conv3 = nnx.Conv(
            in_features=64,
            out_features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            rngs=rngs,
        )
        self.norm3 = nnx.GroupNorm(num_features=64, num_groups=8, rngs=rngs)

        self.fc1 = nnx.Linear(in_features=7 * 7 * 64, out_features=512, rngs=rngs)
        self.head = nnx.Linear(in_features=512, out_features=action_dim, rngs=rngs)

    def __call__(self, x):
        assert x.ndim == 4, "Input must be (batch_size, height, width, channels)"
        x = nnx.relu(self.norm1(self.conv1(x)))
        x = nnx.relu(self.norm2(self.conv2(x)))
        x = nnx.relu(self.norm3(self.conv3(x)))
        x = jnp.reshape(x, (x.shape[0], -1))
        x = nnx.relu(self.fc1(x))
        qvalues = self.head(x)
        return qvalues


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(lz4f.compress(pickle.dumps(data)))

    def sample_batch(self, batch_size: int):
        samples = random.sample(self.buffer, batch_size)
        samples = [pickle.loads(lz4f.decompress(data)) for data in samples]
        states, actions, rewards, next_states, dones = zip(*samples)

        return {
            "states": jnp.array(states),
            "actions": jnp.array(actions),
            "rewards": jnp.array(rewards),
            "next_states": jnp.array(next_states),
            "dones": jnp.array(dones),
        }


@nnx.jit
def train_step(online_network, target_network, data, optimizer, gamma: float = 0.997):

    def loss_fn(online_network):
        q_values = online_network(data["states"])
        q_values_selected = q_values[jnp.arange(len(data["actions"])), data["actions"]]

        next_q_values = target_network(data["next_states"])
        max_next_q_values = jnp.max(next_q_values, axis=1)
        targets = data["rewards"] + gamma * max_next_q_values * (1 - data["dones"])

        # MSE instead of Huber loss for simplicity
        loss = jnp.mean((q_values_selected - targets) ** 2)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(online_network)
    optimizer.update(grads)

    return loss.mean()



def main(env_id: str, outdir: str):

    outdir = Path(outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = utils.get_atari_env(
        env_id, record_folder=outdir / "mp4", record_frequency=250
    )
    action_dim: int = int(env.action_space.n)

    online_network = DQNCNN(action_dim, rngs=nnx.Rngs(0))
    target_network = DQNCNN(action_dim, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(online_network, optax.adam(learning_rate=2e-4))
    replay_buffer = ReplayBuffer(maxlen=1_000_000)

    global_steps, global_episodes = 0, 0
    while global_steps < 5_000_000:
        state, info = env.reset()
        ep_rewards, ep_steps = 0, 0
        lives = info["lives"]

        while True:
            epsilon: float = max(0.1, 1.0 - 0.9 * global_steps / 500_000)  # Epsilon decay
            if epsilon > random.random():
                # Random action (exploration)
                action = env.action_space.sample()
            else:
                # Greedy action (exploitation)
                qvalues = online_network(jnp.expand_dims(state, axis=0))
                action: int = jnp.argmax(qvalues, axis=1).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done: bool = terminated or truncated
            # life loss as episode end
            life_loss: bool = True if lives != info["lives"] else False
            lives = info["lives"]

            replay_buffer.add(state, action, np.clip(reward, -1, 1), next_state, int(life_loss))

            # Update network
            if len(replay_buffer) > 1000 and global_steps % 4 == 0:
                batch_data = replay_buffer.sample_batch(64)
                loss = train_step(online_network, target_network, batch_data, optimizer)
                wandb.log({"loss": loss, "eps": epsilon}, step=global_steps)

            # Sync target network
            if global_steps % 10_000 == 0:
                """Copy weights from online network to target network."""
                nnx.update(target_network, nnx.state(online_network))

            ep_rewards += reward
            ep_steps += 1
            global_steps += 1

            if done:
                print("===="*5)
                print(
                    f"Episode {global_episodes} finished after {ep_steps} steps with reward {ep_rewards}"
                )
                print(f"Global step: {global_steps}")
                wandb.log({"episode_reward": ep_rewards, "episode_steps": ep_steps}, step=global_steps)
                global_episodes += 1
                break

    env.close()


if __name__ == "__main__":
    try:
        wandb.init(project="dqn", mode="disabled")
        main(env_id="Breakout-v4", outdir="out/dqn")
    finally:
        wandb.finish()
