from pathlib import Path
import shutil
import functools
import copy

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

from brax.training.agents.ppo import train as brax_train
from brax.training.agents.ppo import networks as ppo_networks

# from brax import envs as brax_envs


from mujoco_playground import wrapper, registry
from mujoco_playground.config import locomotion_params

# Hyperparameters
NUM_ENVS = 1024
BATCH_SIZE = 256
UNROLL_LENGTH = 20
NUM_UPDATE_PER_BATCH = 32

DISCOUNT = 0.98
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 1.0
CKPT_DIR = "checkpoints"
SEED = 0


def _create_env(env_id: str, num_envs: int = 1, auto_reset: bool = False):
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


def create_env(env_id: str, num_envs: int = 1):
    env, env_cfg = registry.load(env_id), registry.get_default_config(env_id)
    randomizer = registry.get_domain_randomizer(env_id)
    obs_dim: int = env.observation_size["state"][0]
    priv_obs_dim: int = env.observation_size["privileged_state"][0]
    action_dim: int = env.action_size

    if num_envs > 1:
        # all devices gets the same randomization rng
        keys = jax.random.split(jax.random.PRNGKey(42), num_envs)
        v_randomization_fn = functools.partial(randomizer, rng=keys)
        env = wrapper.wrap_for_brax_training(
            env,
            episode_length=1000,
            action_repeat=1,
            randomization_fn=v_randomization_fn,
        )

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    return env, env_cfg, obs_dim, priv_obs_dim, action_dim, reset_fn, step_fn


class SquashedGaussianPolicy(nnx.Module):
    def __init__(self, obs_dim, action_dim: int, rngs: nnx.Rngs):

        self.action_dim = action_dim

        self.dense_1 = nnx.Linear(
            in_features=obs_dim,
            out_features=512,
            kernel_init=nnx.initializers.orthogonal(),
            rngs=rngs,
        )
        self.dense_2 = nnx.Linear(
            in_features=512,
            out_features=256,
            kernel_init=nnx.initializers.orthogonal(),
            rngs=rngs,
        )
        self.dense_3 = nnx.Linear(
            in_features=256,
            out_features=128,
            kernel_init=nnx.initializers.orthogonal(),
            rngs=rngs,
        )

        self.mu = nnx.Linear(
            in_features=128,
            out_features=action_dim,
            kernel_init=nnx.initializers.orthogonal(),
            rngs=rngs,
        )
        self.log_std = nnx.Param(jnp.zeros(action_dim))

    def __call__(self, x):
        x = nnx.elu(self.dense_1(x))
        x = nnx.elu(self.dense_2(x))
        x = nnx.elu(self.dense_3(x))
        mu = self.mu(x)
        std = (nnx.softplus(self.log_std) + 0.001) * jnp.ones_like(mu)
        return mu, std

    @nnx.jit
    def sample_action(self, obs, key: jax.random.PRNGKey):
        assert obs.ndim == 2, "Input must be (batch_size, obs_dim)"
        mu, std = self(obs)
        raw_action = mu + std * jax.random.normal(key, shape=mu.shape)
        action = nnx.tanh(raw_action)
        log_prob = self.log_prob(raw_action, mu, std)
        return action, raw_action, log_prob

    def log_prob(self, raw_action, loc, scale):
        log_prob_normal = -0.5 * (
            jnp.square((raw_action - loc) / scale) + jnp.log(2 * jnp.pi * scale**2)
        ).sum(axis=-1, keepdims=True)

        # log(1 - tanh(x)^2) を数値的に安定した形で計算
        log_det_jacobian = 2.0 * (
            jnp.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action)
        ).sum(axis=-1, keepdims=True)

        log_prob = log_prob_normal - log_det_jacobian
        return log_prob

    def entropy(self, loc, scale, key: jax.random.PRNGKey):
        # tanh(Normal) のエントロピーは解析的に計算できないのでサンプリングして近似
        raw_action_sampled = loc + scale * jax.random.normal(key, shape=loc.shape)

        entropy_normal = 0.5 * (1 + jnp.log(2 * jnp.pi * scale**2)).sum(
            axis=-1, keepdims=True
        )
        # log(1 - tanh(x)^2) を数値的に安定した形で計算
        log_det_jacobian = 2.0 * (
            jnp.log(2.0)
            - raw_action_sampled
            - jax.nn.softplus(-2.0 * raw_action_sampled)
        ).sum(axis=-1, keepdims=True)
        entropy = entropy_normal + log_det_jacobian
        return entropy


class ValueNN(nnx.Module):
    def __init__(self, obs_dim, rngs: nnx.Rngs):

        self.dense_1 = nnx.Linear(
            in_features=obs_dim,
            out_features=512,
            kernel_init=nnx.initializers.orthogonal(),
            rngs=rngs,
        )
        self.dense_2 = nnx.Linear(
            in_features=512,
            out_features=256,
            kernel_init=nnx.initializers.orthogonal(),
            rngs=rngs,
        )
        self.dense_3 = nnx.Linear(
            in_features=256,
            out_features=128,
            kernel_init=nnx.initializers.orthogonal(),
            rngs=rngs,
        )
        self.out = nnx.Linear(
            in_features=128,
            out_features=1,
            kernel_init=nnx.initializers.orthogonal(),
            rngs=rngs,
        )

    def __call__(self, x):
        x = nnx.elu(self.dense_1(x))
        x = nnx.elu(self.dense_2(x))
        x = nnx.elu(self.dense_3(x))
        out = self.out(x)
        return out


@nnx.jit
def train_step(
    batch_data: dict,
    policy_nn: SquashedGaussianPolicy,
    value_nn: ValueNN,
    policy_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
    key: jax.random.PRNGKey,
):
    advantages = batch_data["advantages"]
    normalized_advantages = (advantages - jnp.mean(advantages)) / (
        jnp.std(advantages) + 1e-8
    )

    def policy_loss_fn(policy_nn):
        mu, std = policy_nn(batch_data["obs_policy"])
        new_log_probs = policy_nn.log_prob(batch_data["raw_actions"], loc=mu, scale=std)

        ratio = jnp.exp(new_log_probs - batch_data["old_log_probs"])
        surr1 = ratio * normalized_advantages
        surr2 = jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * normalized_advantages
        policy_loss = -1 * jnp.minimum(surr1, surr2)

        entropy = policy_nn.entropy(mu, std, key)
        entropy_loss = -1 * ENTROPY_COEF * entropy

        loss = jnp.mean(policy_loss + entropy_loss)

        return loss

    def value_loss_fn(value_nn):
        values = value_nn(batch_data["obs_value"])
        value_loss = (values - batch_data["target_values"]) ** 2
        loss = jnp.mean(value_loss)
        return loss

    p_loss, p_grad = nnx.value_and_grad(policy_loss_fn)(policy_nn)
    policy_optimizer.update(p_grad)

    v_loss, v_grad = nnx.value_and_grad(value_loss_fn)(value_nn)
    value_optimizer.update(v_grad)

    return p_loss, v_loss


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


def train(env_id: str, log_dir: str):

    log_dir = Path(log_dir)
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    (env, env_cfg, obs_dim, priv_obs_dim, action_dim, env_reset_fn, env_step_fn) = (
        create_env(env_id, num_envs=NUM_ENVS)
    )
    # ppo_config = locomotion_params.brax_ppo_config(env_id)

    policy_nn = SquashedGaussianPolicy(
        obs_dim=obs_dim, action_dim=action_dim, rngs=nnx.Rngs(0)
    )
    policy_optimizer = nnx.Optimizer(
        policy_nn,
        optax.chain(
            optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(learning_rate=3e-4)
        ),
    )

    value_nn = ValueNN(obs_dim=priv_obs_dim, rngs=nnx.Rngs(0))
    value_optimizer = nnx.Optimizer(
        value_nn,
        optax.chain(
            optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(learning_rate=3e-4)
        ),
    )

    rng, *subkeys = jax.random.split(jax.random.PRNGKey(SEED), NUM_ENVS + 1)
    state = env_reset_fn(jnp.array(subkeys))
    trajectory, selected_actions = [state], []
    for i in tqdm(range(1, 200_000_000 // NUM_ENVS)):
        rng, subkey = jax.random.split(rng)
        action, raw_action, log_prob = policy_nn.sample_action(
            state.obs["state"], subkey
        )
        selected_actions.append((action, raw_action, log_prob))

        state = env_step_fn(state, action)
        trajectory.append(state)

        if i % UNROLL_LENGTH == 0:
            assert len(trajectory) == UNROLL_LENGTH + 1
            assert len(selected_actions) == UNROLL_LENGTH

            rewards = jnp.stack([s.reward for s in trajectory[1:]], axis=1)
            dones = jnp.stack([s.done for s in trajectory[1:]], axis=1)

            advantages, target_values = compute_advantage_and_target(
                value_nn,
                obs=jnp.stack([s.obs["privileged_state"] for s in trajectory], axis=1),
                rewards=rewards,
                dones=dones,
            )

            B, T = NUM_ENVS, UNROLL_LENGTH
            batch_data = {
                "obs_policy": jnp.stack(
                    [s.obs["state"] for s in trajectory[:-1]], axis=1
                ).reshape(B * T, -1),
                "obs_value": jnp.stack(
                    [s.obs["privileged_state"] for s in trajectory[:-1]], axis=1
                ).reshape(B * T, -1),
                "raw_actions": jnp.stack(
                    [a[1] for a in selected_actions], axis=1
                ).reshape(B * T, -1),
                "old_log_probs": jnp.stack(
                    [a[2] for a in selected_actions], axis=1
                ).reshape(B * T, 1),
                "advantages": advantages.reshape(B * T, 1),
                "target_values": target_values.reshape(B * T, 1),
            }

            # Update networks
            for _ in range(NUM_UPDATE_PER_BATCH):
                rng, subkey1, subkey2 = jax.random.split(rng, 3)
                indices = jax.random.permutation(subkey1, B * T)[:BATCH_SIZE]
                _batch_data = jax.tree_util.tree_map(lambda x: x[indices], batch_data)

                ploss, vloss = train_step(
                    batch_data=_batch_data,
                    policy_nn=policy_nn,
                    value_nn=value_nn,
                    policy_optimizer=policy_optimizer,
                    value_optimizer=value_optimizer,
                    key=subkey2,
                )
            else:
                wandb.log(
                    {
                        "ploss": ploss,
                        "vloss": vloss,
                        "reward": rewards.sum(axis=-1).mean(),
                    },
                    step=i * NUM_ENVS,
                )

            trajectory = trajectory[-1:]  # Keep the last state for the next rollout
            selected_actions = []  # Reset actions for the next rollout

        if i % 10_000 == 0:
            # Save the model checkpoint
            checkpointer = ocp.StandardCheckpointer()
            ckpt_dir: Path = Path(log_dir / CKPT_DIR).resolve()
            _, _state = nnx.split(policy_nn)
            checkpointer.save(ckpt_dir, _state, force=True)

            # Evaluate
            test_score = evaluate(
                env_id=env_id,
                n_episodes=3,
                log_dir=log_dir,
                record_video=True if i % 40_000 == 0 else False,
            )
            wandb.log(
                {"episode_reward": test_score},
                step=i * NUM_ENVS,
            )


def evaluate(
    env_id: str,
    n_episodes: int,
    log_dir: str,
    record_video: bool = True,
):

    (env, env_cfg, obs_dim, _, action_dim, env_reset_fn, env_step_fn) = create_env(
        env_id
    )

    # Load the trained policy
    abstract_model = nnx.eval_shape(
        lambda: SquashedGaussianPolicy(
            obs_dim=obs_dim, action_dim=action_dim, rngs=nnx.Rngs(0)
        )
    )
    checkpointer = ocp.StandardCheckpointer()
    ckpt_dir: Path = (Path(log_dir) / CKPT_DIR).resolve()
    _graphdef, _abstract_state = nnx.split(abstract_model)
    _state = checkpointer.restore(ckpt_dir, _abstract_state)
    policy_nn = nnx.merge(_graphdef, _state)

    scores = []
    for n in range(n_episodes):
        # print(f"Evaluating episode {n + 1}/{n_episodes}...")
        rng = jax.random.PRNGKey(n)
        state = env_reset_fn(rng)
        trajectory = [state]
        for _ in range(env_cfg.episode_length):
            rng, subkey = jax.random.split(rng)
            action, _, _ = policy_nn.sample_action(
                state.obs["state"].reshape(1, -1), subkey
            )
            state = env_step_fn(state, action[0])
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


import click


@click.group()
def cli():
    pass


@cli.command(name="train")
@click.option("--env-id", default="Go1JoystickFlatTerrain", help="Environment ID")
@click.option("--log-dir", default="log", help="Directory to save logs and videos")
@click.option("--use-wandb", is_flag=True, help="Enable wandb (default: disable)")
def run_training(env_id: str, log_dir: str, use_wandb: bool):
    try:
        wandb.init(
            project="ppo",
            mode="online" if use_wandb else "disabled",
        )
        train(env_id=env_id, log_dir=f"{log_dir}/{env_id}")
    finally:
        wandb.finish()


@cli.command(name="eval")
@click.option("--env-id", default="Go1JoystickFlatTerrain", help="Environment ID")
@click.option("--log-dir", default="log", help="Directory to save logs and videos")
def run_evaluation(env_id: str, log_dir: str):
    evaluate(
        env_id=env_id, log_dir=f"{log_dir}/{env_id}", n_episodes=5, record_video=True
    )


@cli.command(name="dev")
def debug():
    pass
    import pdb; pdb.set_trace()  # fmt: skip


if __name__ == "__main__":
    cli()
