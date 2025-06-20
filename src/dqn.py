from typing import Literal
import functools

import jax
import jax.numpy as jnp
from flax import nnx

from src.tools.envs import get_atari_env, ENV_INFO



class DQNCNN(nnx.Module):
    def __init__(self):
        ...
    def __call__(self):
        ...
    def sample_action(self, epsilon: float = 0.0):
        ...


class ReplayBuffer:
    def __init__(self, maxlen: int):
        ...
    def add(self):
        ...
    def sample_minibatch(self):
        ...


def linear_scheduler():
    ...

def update_network(online_network, target_network, data):
    ...

def sync_network(online_network, target_network):
    ...

def main(env_id: Literal["Breakout", "CartPole"]):
    online_network, target_network = DQNCNN(), DQNCNN()
    replay_buffer = ReplayBuffer(maxlen=100_000)
    eps_schduler = functools.partial(linear_scheduler)

    n = 0
    while n < 1_000_000:
        env, obs = get_atari_env(env_id)
        done = False
        while not done:
            action = DQNCNN.sample_action(eps=eps_schduler(n))
            env.step(action)
            if n > 1000 and n % 4 == 0:
                minibatch = replay_buffer.sample_minibatch()
                update_network(online_network, target_network, minibatch)

            if n % 400 == 0:
                sync_network(online_network, target_network)



if __name__ == "__main__":
    main(env_id="Breakout")
