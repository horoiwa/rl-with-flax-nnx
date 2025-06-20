from typing import Literal
import functools
import random
from collections import deque

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import optax

from src.tools.envs import get_atari_env, ENV_INFO



class DQNCNN(nnx.Module):
    def __init__(self, action_dim: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_features=4, out_features=32, kernel_size=(8, 8), strides=(4, 4), rngs=rngs)
        self.conv2 = nnx.Conv(in_features=32, out_features=64, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
        self.conv3 = nnx.Conv(in_features=64, out_features=64, kernel_size=(3, 3), strides=(1, 1), rngs=rngs)

        # Calculate the correct input size for dense layer
        dummy_input = jnp.ones((1, 84, 84, 4))
        conv_output = self._conv_forward(dummy_input)
        flattened_size = conv_output.shape[1]

        self.dense1 = nnx.Linear(in_features=flattened_size, out_features=512, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=512, out_features=action_dim, rngs=rngs)

    def _conv_forward(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        return x.reshape(x.shape[0], -1)

    def __call__(self, x):
        x = self._conv_forward(x)
        x = nnx.relu(self.dense1(x))
        return self.dense2(x)

    def sample_action(self, state, epsilon: float = 0.0, rng_key=None):
        if rng_key is None:
            rng_key = jax.random.PRNGKey(random.randint(0, 2**32 - 1))

        if jax.random.uniform(rng_key) < epsilon:
            return jax.random.randint(rng_key, (), 0, self.dense2.out_features)
        else:
            q_values = self(state)
            return jnp.argmax(q_values, axis=-1)[0]  # Take first element for scalar


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_minibatch(self, batch_size: int = 32):
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return {
            'states': jnp.array(states),
            'actions': jnp.array(actions),
            'rewards': jnp.array(rewards),
            'next_states': jnp.array(next_states),
            'dones': jnp.array(dones)
        }

    def __len__(self):
        return len(self.buffer)


def linear_scheduler(step: int, start_eps: float = 1.0, end_eps: float = 0.01, decay_steps: int = 100000):
    """Linear epsilon decay scheduler."""
    if step >= decay_steps:
        return end_eps
    return start_eps - (start_eps - end_eps) * (step / decay_steps)

def update_network(online_network, target_network, data, optimizer, gamma: float = 0.99):
    """Update the online network using DQN loss."""
    def loss_fn(online_network):
        q_values = online_network(data['states'])
        q_values_selected = q_values[jnp.arange(len(data['actions'])), data['actions']]

        next_q_values = target_network(data['next_states'])
        max_next_q_values = jnp.max(next_q_values, axis=1)

        targets = data['rewards'] + gamma * max_next_q_values * (1 - data['dones'])

        loss = jnp.mean((q_values_selected - targets) ** 2)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(online_network)
    optimizer.update(grads)
    return loss

def sync_network(online_network, target_network):
    """Copy weights from online network to target network."""
    state = nnx.state(online_network)
    nnx.update(target_network, state)

def main(env_id: Literal["Breakout", "CartPole"]):
    # Initialize environment
    env = get_atari_env(env_id)
    action_dim = ENV_INFO[env_id]["action_space"]

    # Initialize networks
    rngs = nnx.Rngs(0)
    online_network = DQNCNN(action_dim, rngs)
    target_network = DQNCNN(action_dim, rngs)
    sync_network(online_network, target_network)

    # Initialize optimizer and buffer
    optimizer = nnx.Optimizer(online_network, optax.adam(learning_rate=1e-4))
    replay_buffer = ReplayBuffer(maxlen=100_000)

    # Training parameters
    rng_key = jax.random.PRNGKey(42)
    episode_rewards = []

    for episode in range(1000):
        obs, _ = env.reset()
        obs_stack = deque([obs] * 4, maxlen=4)  # Stack 4 frames
        episode_reward = 0
        step = 0

        while True:
            # Get current state (stacked frames)
            state = jnp.array(obs_stack).transpose(1, 2, 0)
            state = jnp.expand_dims(state, 0)  # Add batch dimension

            # Sample action
            rng_key, subkey = jax.random.split(rng_key)
            epsilon = linear_scheduler(episode * 1000 + step)
            action = online_network.sample_action(state, epsilon, subkey)
            action = int(action)

            # Take step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update observation stack
            obs_stack.append(next_obs)
            next_state = jnp.array(obs_stack).transpose(1, 2, 0)

            # Store transition
            replay_buffer.add(
                state.squeeze(0), action, reward, next_state, done
            )

            episode_reward += reward
            step += 1

            # Training
            if len(replay_buffer) > 1000 and step % 4 == 0:
                minibatch = replay_buffer.sample_minibatch(32)
                if minibatch is not None:
                    loss = update_network(online_network, target_network, minibatch, optimizer)

            # Sync target network
            if step % 1000 == 0:
                sync_network(online_network, target_network)

            if done:
                break

        episode_rewards.append(episode_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}, Average Reward: {np.mean(episode_rewards[-100:]):.2f}")

    env.close()



if __name__ == "__main__":
    main(env_id="Breakout")
