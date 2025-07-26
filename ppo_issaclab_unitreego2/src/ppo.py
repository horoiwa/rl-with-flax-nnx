from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
import orbax.checkpoint as ocp
import wandb


class GaussianPolicy(nnx.Module):
    def __init__(self, obs_dim, action_dim: int, rngs: nnx.Rngs):

        self.action_dim = action_dim

        self.dense_1 = nnx.Linear(in_features=obs_dim, out_features=128, rngs=rngs)
        self.dense_2 = nnx.Linear(in_features=128, out_features=128, rngs=rngs)
        self.dense_3 = nnx.Linear(in_features=128, out_features=128, rngs=rngs)

        self.mu = nnx.Linear(in_features=128, out_features=action_dim, rngs=rngs)
        self.log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, x):
        x = nnx.elu(self.dense_1(x))
        x = nnx.elu(self.dense_2(x))
        x = nnx.elu(self.dense_3(x))
        mu = nnx.tanh(self.mu(x))
        std = jnp.exp(self.log_std)
        return mu, std

    def sample_action(self, obs, key: jax.random.Key):
        mu, std = self(obs)
        action = mu + std * jax.random.normal(key, shape=(self.action_dim,))
        return action


class Value(nnx.Module):
    def __init__(self, obs_dim, rngs: nnx.Rngs):

        self.dense_1 = nnx.Linear(in_features=obs_dim, out_features=128, rngs=rngs)
        self.dense_2 = nnx.Linear(in_features=128, out_features=128, rngs=rngs)
        self.dense_3 = nnx.Linear(in_features=128, out_features=128, rngs=rngs)
        self.out = nnx.Linear(in_features=128, out_features=1, rngs=rngs)

    def __call__(self, x):
        x = nnx.elu(self.dense_1(x))
        x = nnx.elu(self.dense_2(x))
        x = nnx.elu(self.dense_3(x))
        out = self.out(x)
        return out


# @nnx.jit
def train_step(
    data,
    policy_network,
    policy_optimizer,
    value_network,
    value_optimizer,
):

    def policy_loss_fn(policy_network):
        loss = 0.0
        return loss

    def value_loss_fn(policy_network):
        loss = 0.0
        return loss

    ploss, pgrads = nnx.value_and_grad(policy_loss_fn)(policy_network)
    policy_optimizer.update(pgrads)

    vloss, vgrads = nnx.value_and_grad(value_loss_fn)(value_network)
    value_optimizer.update(vgrads)

    return ploss.mean(), vloss.mean()


def compute_advantage(data: dict):
    return data


def main(env_id: str, outdir: str):

    outdir = Path(outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import pdb; pdb.set_trace()  # fmt: skip
    env = None
    action_dim: int = int(env.action_space.n)

    policy_nn = GaussianPolicy()
    value_nn = Value()


if __name__ == "__main__":
    try:
        wandb.init(project="ppo", mode="disabled")
        main(env_id="Isaac-Velocity-Flat-Unitree-Go2-v0", outdir="log")
    finally:
        wandb.finish()
