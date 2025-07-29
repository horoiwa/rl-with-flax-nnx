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

    # @nnx.jit
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
    batch_data: dict,
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


# @nnx.jit
def compute_advantage(rewards, dones, value_network, gamma=0.99):
    """Computes advantage(GAE)"""
    pass


def train(env_id: str, num_envs: int, log_dir: str):
    log_dir = Path(log_dir)
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    (env, env_cfg, obs_dim, action_dim, reset_fn, step_fn) = create_env(
        env_id, num_envs=num_envs, auto_reset=True
    )
    ppo_params = locomotion_params.brax_ppo_config(env_id)

    policy_nn = GaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, rngs=nnx.Rngs(0))
    policy_optimizer = optax.adam(learning_rate=1e-3)

    value_nn = ValueNN(obs_dim=obs_dim, rngs=nnx.Rngs(0))
    value_optimizer = optax.adam(learning_rate=1e-3)

    rng, *keys = jax.random.split(jax.random.PRNGKey(0), num_envs + 1)
    states = reset_fn(jnp.array(keys))

    trajcetory = [states]
    for i in range(100_000_000 // num_envs):
        rng, key = jax.random.split(rng)
        actions = policy_nn.sample_action(states.obs["state"], key)
        states = step_fn(states, actions)
        trajcetory.append(states)
        i += num_envs

        if i % 4 == 0:
            _trajectory, trajectory = trajectory[:5], trajcetory[4:]
            batch_data = {
                "obs": jnp.stack([s.obs["state"] for s in _trajectory]),
                "actions": jnp.stack([s.action for s in _trajectory]),
                "rewards": jnp.stack([s.reward for s in _trajectory]),
                "dones": jnp.stack([s.done for s in _trajectory]),
            }
            batch_data["advantages"] = compute_advantage(
                batch_data["obs"], batch_data["rewards"], batch_data["dones"], value_nn
            )

            train_step(
                batch_data,
                policy_nn,
                policy_optimizer,
                value_nn,
                value_optimizer,
            )

        if i % 1000 == 0:
            score = evaluate(
                env_id=env_id, n_episodes=5, log_dir=log_dir, record_video=False
            )


def evaluate(env_id: str, n_episodes: int, log_dir: str, record_video: bool = True):

    (env, env_cfg, obs_dim, action_dim, env_reset_fn, env_step_fn) = create_env(env_id)

    policy_nn = GaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, rngs=nnx.Rngs(0))

    scores = []
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

        score: float = sum([s.reward for s in trajectory])
        scores.append(score)
        print(f"Episode {n + 1}, {len(trajectory)} steps, total reward: {score:.2f}")

        if record_video:
            print("Saving video...")
            frames: list[np.ndarray] = env.render(trajectory, camera="track")
            imageio.mimsave(f"{log_dir}/eval_{n+1}.mp4", frames, fps=1 / env.dt)

    return np.mean(scores)


if __name__ == "__main__":
    try:
        # wandb.init(project="ppo", mode="disabled")
        train(env_id="Go1JoystickFlatTerrain", num_envs=4, outdir="log")
        # evaluate(env_id="Go1JoystickFlatTerrain", outdir="log", n_episodes=5)
    finally:
        wandb.finish()
