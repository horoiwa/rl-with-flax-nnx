from pathlib import Path
import shutil

import wandb

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
import orbax.checkpoint as ocp
from brax.training.agents.ppo import train


from mujoco_playground import wrapper, registry
from mujoco_playground.config import locomotion_params


class GaussianPolicy(nnx.Module):
    def __init__(self, obs_dim, action_dim: int, rngs: nnx.Rngs):

        self.action_dim = action_dim

        self.dense_1 = nnx.Linear(in_features=obs_dim, out_features=512, rngs=rngs)
        self.dense_2 = nnx.Linear(in_features=512, out_features=256, rngs=rngs)
        self.dense_3 = nnx.Linear(in_features=256, out_features=128, rngs=rngs)

        self.mu = nnx.Linear(in_features=128, out_features=action_dim, rngs=rngs)
        self.log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, x):
        x = nnx.elu(self.dense_1(x))
        x = nnx.elu(self.dense_2(x))
        x = nnx.elu(self.dense_3(x))
        mu = nnx.tanh(self.mu(x))
        std = jnp.exp(self.log_std)
        return mu, std

    def sample_action(self, obs, key: nnx.Rngs):
        mu, std = self(obs)
        action = mu + std * jax.random.normal(key, shape=(self.action_dim,))
        return action


class Value(nnx.Module):
    def __init__(self, obs_dim, rngs: nnx.Rngs):

        self.dense_1 = nnx.Linear(in_features=obs_dim, out_features=128, rngs=rngs)
        self.dense_2 = nnx.Linear(in_features=512, out_features=256, rngs=rngs)
        self.dense_3 = nnx.Linear(in_features=256, out_features=128, rngs=rngs)
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


def main(env_id: str, num_envs: int, outdir: str):

    outdir = Path(outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = registry.load(env_id)
    reset_env_fn = jax.vmap(env.reset, in_axes=(0, None))
    step_env_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    rng = jax.random.PRNGKey(0)
    rng, *keys = jax.random.split(rng, num_envs + 1)
    keys = jnp.stack(keys)

    import pdb; pdb.set_trace()  # fmt: skip

    states = env.reset(key)
    action_dim: int = int(env.action_space.n)

    policy_nn = GaussianPolicy()
    value_nn = Value()
    # ppo_params = locomotion_params.brax_ppo_config(env_id)


if __name__ == "__main__":
    try:
        wandb.init(project="ppo", mode="disabled")
        # main(env_id="Go1Footstand", outdir="log")
        main(env_id="Go1JoystickFlatTerrain", num_envs=12, outdir="log")
    finally:
        wandb.finish()
