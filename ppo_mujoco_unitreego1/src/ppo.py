from pathlib import Path
import shutil

import wandb

import imageio
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
import orbax.checkpoint as ocp

# from brax.training.agents.ppo import train as brax_train
# from brax import envs as brax_envs


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

    def sample_action(self, obs, key: jax.random.PRNGKey):
        mu, std = self(obs)
        action = mu + std * jax.random.normal(key, shape=(self.action_dim,))
        return action


class ValueNN(nnx.Module):
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
    def compute_advantage(data: dict):
        return data

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


def create_env(env_id: str, num_envs: int = 1, auto_reset: bool = False):
    env, env_cfg = registry.load(env_id), registry.get_default_config(env_id)
    obs_dim: int = env.observation_size["state"][0]
    action_dim: int = env.action_size
    if auto_reset:
        env = wrapper.BraxAutoResetWrapper(env)

    if num_envs == 1:
        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)
    else:
        reset_fn = jax.jit(jax.vmap(env.reset, in_axes=(0,)))
        step_fn = jax.jit(jax.vmap(env.step, in_axes=(0, 0)))

    return env, env_cfg, obs_dim, action_dim, reset_fn, step_fn


def train(env_id: str, num_envs: int, outdir: str):
    outdir = Path(outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(0)

    (env, env_cfg, obs_dim, action_dim, reset_fn, step_fn) = create_env(
        env_id, num_envs=num_envs, auto_reset=True
    )

    policy_nn = GaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, rngs=nnx.Rngs(0))
    value_nn = ValueNN(obs_dim=obs_dim, rngs=nnx.Rngs(0))

    rng, keys = jax.random.split(rng)
    states = reset_fn(jax.random.split(keys, num_envs))
    while (i := 0) <= 100_000_000:
        trajectory = []
        for _ in range(5):
            i += num_envs


def evaluate(env_id: str, n_episodes: int, outdir: str):

    (env, env_cfg, obs_dim, action_dim, env_reset_fn, env_step_fn) = create_env(env_id)

    policy_nn = GaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, rngs=nnx.Rngs(0))

    for n in tqdm(range(n_episodes)):
        print(f"Evaluating episode {n + 1}/{n_episodes}...")
        rng = jax.random.PRNGKey(n)
        state = env_reset_fn(rng)
        trajectory = [state]
        for _ in range(env_cfg.episode_length):
            rng, subkey = jax.random.split(rng)
            action = policy_nn.sample_action(state.obs["state"], subkey)
            state = env_step_fn(state, action)
            trajectory.append(state)
            if state.done:
                break

        total_reward: float = sum([s.reward for s in trajectory])
        print(
            f"Episode {n + 1}, {len(trajectory)} steps, total reward: {total_reward:.2f}"
        )
        print("Saving video...")
        frames: list[np.ndarray] = env.render(trajectory, camera="track")
        imageio.mimsave(f"{outdir}/eval_{n+1}.mp4", frames, fps=1 / env.dt)


# ppo_params = locomotion_params.brax_ppo_config(env_id)

if __name__ == "__main__":
    try:
        # wandb.init(project="ppo", mode="disabled")
        train(env_id="Go1JoystickFlatTerrain", num_envs=4, outdir="log")
        # evaluate(env_id="Go1JoystickFlatTerrain", outdir="log", n_episodes=5)
    finally:
        wandb.finish()
