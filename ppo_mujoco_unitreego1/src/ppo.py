from pathlib import Path
import shutil
import functools

import wandb

import imageio
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.scipy.stats
import numpy as np
from flax import nnx
import optax
import orbax.checkpoint as ocp

# from brax.training.agents.ppo import train as brax_train
# from brax import envs as brax_envs


from mujoco_playground import wrapper, registry
from mujoco_playground.config import locomotion_params

# Hyperparameters
UNROLL_LENGTH = 20
BATCH_SIZE = 16  # 256
NUM_UPDATE_PER_BATCH = 4
DISCOUNT = 0.98
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COST = 0.01
MAX_GRAD_NORM = 1.0


def create_env(env_id: str, num_envs: int = 1, auto_reset: bool = False):
    env, env_cfg = registry.load(env_id), registry.get_default_config(env_id)
    obs_dim: int = env.observation_size["state"][0]
    priv_obs_dim: int = env.observation_size["privileged_state"][0]
    action_dim: int = env.action_size
    if auto_reset:
        env = wrapper.BraxAutoResetWrapper(env)

    if num_envs == 1:
        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)
    else:
        reset_fn = jax.jit(jax.vmap(env.reset, in_axes=(0,)))
        step_fn = jax.jit(jax.vmap(env.step, in_axes=(0, 0)))

    return env, env_cfg, obs_dim, priv_obs_dim, action_dim, reset_fn, step_fn


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
        log_prob = jax.scipy.stats.norm.logpdf(action, loc=mu, scale=std).sum(axis=-1)
        return action, log_prob


class ValueNN(nnx.Module):
    def __init__(self, obs_dim, rngs: nnx.Rngs):

        self.dense_1 = nnx.Linear(in_features=obs_dim, out_features=512, rngs=rngs)
        self.dense_2 = nnx.Linear(in_features=512, out_features=256, rngs=rngs)
        self.dense_3 = nnx.Linear(in_features=256, out_features=128, rngs=rngs)
        self.out = nnx.Linear(in_features=128, out_features=1, rngs=rngs)

    @nnx.jit
    def __call__(self, x):
        x = nnx.elu(self.dense_1(x))
        x = nnx.elu(self.dense_2(x))
        x = nnx.elu(self.dense_3(x))
        out = self.out(x)
        return out


def train_step(
    batch_data: dict,
    key: jax.random.PRNGKey,
    policy_nn: GaussianPolicy,
    value_nn: ValueNN,
    policy_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
):
    """Performs one epoch of PPO updates."""
    B = batch_data["obs_policy"].shape[0] * batch_data["obs_policy"].shape[1]

    flat_batch = {
        "obs_policy": batch_data["obs_policy"].reshape(B, -1),
        "obs_value": batch_data["obs_value"][:, :-1].reshape(B, -1),
        "actions": batch_data["actions"].reshape(B, -1),
        "log_probs_old": batch_data["log_probs"].reshape(B),
        "advantages": batch_data["advantages"].reshape(B),
        "target_values": batch_data["target_values"].reshape(B),
    }

    # Normalize advantages
    advantages = (flat_batch["advantages"] - jnp.mean(flat_batch["advantages"])) / (
        jnp.std(flat_batch["advantages"]) + 1e-8
    )
    flat_batch["normalized_advantages"] = advantages

    # Shuffle data
    key, subkey = jax.random.split(key)
    permutation = jax.random.permutation(subkey, B)
    shuffled_batch = jax.tree_util.tree_map(lambda x: x[permutation], flat_batch)

    # Split into minibatches
    minibatch_size = B // BATCH_SIZE
    minibatches = jax.tree_util.tree_map(
        lambda x: x.reshape((BATCH_SIZE, minibatch_size) + x.shape[1:]),
        shuffled_batch,
    )

    def update_minibatch(carry, minibatch_data):
        policy_opt_state, value_opt_state = carry

        def policy_loss_fn(params):
            policy_nn.update(params)
            mu, std = policy_nn(minibatch_data["obs_policy"])
            new_log_probs = jax.scipy.stats.norm.logpdf(
                minibatch_data["actions"], loc=mu, scale=std
            ).sum(axis=-1)

            ratio = jnp.exp(new_log_probs - minibatch_data["log_probs_old"])
            surr1 = ratio * minibatch_data["advantages"]
            surr2 = (
                jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                * minibatch_data["advantages"]
            )
            policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

            entropy = jnp.mean(
                jax.scipy.stats.norm.entropy(loc=mu, scale=std).sum(axis=-1)
            )
            entropy_loss = ENTROPY_COST * -entropy

            return policy_loss + entropy_loss

        def value_loss_fn(params):
            value_nn.update(params)
            predicted_values = value_nn(minibatch_data["obs_value"]).squeeze()
            return jnp.mean((predicted_values - minibatch_data["target_values"]) ** 2)

        # Update policy
        p_grad = nnx.grad(policy_loss_fn, wrt=nnx.Param)(policy_nn.get(nnx.Param))
        p_updates, new_policy_opt_state = policy_optimizer.update(
            p_grad, policy_opt_state, policy_nn.get(nnx.Param)
        )
        policy_nn.update(p_updates)

        # Update value
        v_grad = nnx.grad(value_loss_fn, wrt=nnx.Param)(value_nn.get(nnx.Param))
        v_updates, new_value_opt_state = value_optimizer.update(
            v_grad, value_opt_state, value_nn.get(nnx.Param)
        )
        value_nn.update(v_updates)

        p_loss = policy_loss_fn(policy_nn.get(nnx.Param))
        v_loss = value_loss_fn(value_nn.get(nnx.Param))

        return (new_policy_opt_state, new_value_opt_state), (p_loss, v_loss)

    (policy_opt_state, value_opt_state), (p_losses, v_losses) = jax.lax.scan(
        update_minibatch, (policy_opt_state, value_opt_state), minibatches
    )

    return jnp.mean(p_losses), jnp.mean(v_losses), policy_opt_state, value_opt_state


@nnx.jit
def compute_advantage_and_target(value_nn, obs, rewards, dones):
    """Computes advantage(GAE) and value targets."""
    B, T, _ = obs.shape
    values = value_nn(obs.reshape(B * T, -1)).reshape(B, T)
    values_t, values_t_plus_1 = values[:, :-1], values[:, 1:]
    deltas = rewards + DISCOUNT * (1 - dones) * values_t_plus_1 - values_t

    def gae_scan_fn(advantage_plus_1, data_t):
        delta_t, done_t = data_t
        advantage_t = delta_t + DISCOUNT * GAE_LAMBDA * (1 - done_t) * advantage_plus_1
        return advantage_t, advantage_t

    initial_carry = jnp.zeros(B)
    _, gae_T = jax.lax.scan(
        gae_scan_fn, initial_carry, (deltas.T, dones.T), reverse=True
    )
    gae = gae_T.T
    target_values = gae + values_t
    return gae, target_values


def train(env_id: str, num_envs: int, log_dir: str):
    log_dir = Path(log_dir)
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    (env, env_cfg, obs_dim, priv_obs_dim, action_dim, env_reset_fn, env_step_fn) = (
        create_env(env_id, num_envs=num_envs, auto_reset=True)
    )
    # ppo_config = locomotion_params.brax_ppo_config(env_id)

    policy_nn = GaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, rngs=nnx.Rngs(0))
    policy_optimizer = optax.adam(learning_rate=3e-4)

    value_nn = ValueNN(obs_dim=priv_obs_dim, rngs=nnx.Rngs(0))
    value_optimizer = optax.adam(learning_rate=3e-4)

    rng, *subkeys = jax.random.split(jax.random.PRNGKey(0), num_envs + 1)
    state = env_reset_fn(jnp.array(subkeys))
    trajectory, selected_actions = [state], []
    for i in range(1, 100_000_000 // num_envs):
        rng, *subkeys = jax.random.split(rng, num_envs + 1)
        action, log_prob = jax.vmap(policy_nn.sample_action)(
            state.obs["state"], jnp.array(subkeys)
        )
        selected_actions.append((action, log_prob))

        state = env_step_fn(state, action)
        trajectory.append(state)

        if i % UNROLL_LENGTH == 0:
            assert len(trajectory) == UNROLL_LENGTH + 1
            # ここでいきなりbatch_dictをつくらず、さいごにフラットなバッチをつくる
            batch_data = {
                "obs_policy": jnp.stack(
                    [s.obs["state"] for s in trajectory[:-1]], axis=1
                ),
                "obs_value": jnp.stack(
                    [s.obs["privileged_state"] for s in trajectory], axis=1
                ),
                "actions": jnp.stack([a[0] for a in selected_actions], axis=1),
                "log_probs": jnp.stack([a[1] for a in selected_actions], axis=1),
                "rewards": jnp.stack([s.reward for s in trajectory[1:]], axis=1),
                "dones": jnp.stack([s.done for s in trajectory[1:]], axis=1),
            }

            advantages, target_values = compute_advantage_and_target(
                value_nn,
                obs=batch_data["obs_value"],
                rewards=batch_data["rewards"],
                dones=batch_data["dones"],
            )

            batch_data = {
                "obs": None,
                "actions": None,
                "log_probs": None,
                "advantages": advantages,
                "target_values": target_values,
            }

            # Update networks
            for _ in range(NUM_UPDATE_PER_BATCH):
                rng, subkey = jax.random.split(rng)
                ploss, vloss = train_step(
                    batch_data=batch_data,
                    rng=subkey,
                    policy_nn=policy_nn,
                    value_nn=value_nn,
                    policy_optimizer=policy_optimizer,
                    value_optimizer=value_optimizer,
                )

            trajectory = [trajectory[-1]]  # Keep the last state for the next rollout
            selected_actions = []  # Reset actions for the next rollout

        if i % 1000 == 0:
            print(f"Step {i}: policy_loss={ploss:.4f}, value_loss={vloss:.4f}")
            test_score = evaluate(
                env_id=env_id, n_episodes=5, log_dir=log_dir, record_video=False
            )


def evaluate(env_id: str, n_episodes: int, log_dir: str, record_video: bool = True):

    (env, env_cfg, obs_dim, _, action_dim, env_reset_fn, env_step_fn) = create_env(
        env_id
    )

    policy_nn = GaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, rngs=nnx.Rngs(0))

    scores = []
    for n in tqdm(range(n_episodes)):
        print(f"Evaluating episode {n + 1}/{n_episodes}...")
        rng = jax.random.PRNGKey(n)
        state = env_reset_fn(rng)
        trajectory = [state]
        for _ in range(env_cfg.episode_length):
            rng, subkey = jax.random.split(rng)
            action, _ = policy_nn.sample_action(state.obs["state"], subkey)
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
        train(env_id="Go1JoystickFlatTerrain", num_envs=4, log_dir="log")
        # evaluate(env_id="Go1JoystickFlatTerrain", log_dir="log", n_episodes=5)
    finally:
        wandb.finish()
