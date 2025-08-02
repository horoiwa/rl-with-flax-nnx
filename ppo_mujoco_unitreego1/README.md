
# RUN

`python ppo.py train --use-wandb`

`python ppo.py eval`


# Reference

(mujoco-mjx)[https://github.com/google-deepmind/mujoco/tree/main/mjx]


(mujoco-playground)[https://playground.mujoco.org/]

(unitree-go1)[https://github.com/google-deepmind/mujoco_menagerie/blob/main/unitree_go1/go1.xml]


(Pdb) ppo_params
action_repeat: 1
batch_size: 256
discounting: 0.97
entropy_cost: 0.01
episode_length: 1000
learning_rate: 0.0003
max_grad_norm: 1.0
network_factory:
  policy_hidden_layer_sizes: &id001 !!python/tuple
  - 512
  - 256
  - 128
  policy_obs_key: state
  value_hidden_layer_sizes: *id001
  value_obs_key: privileged_state
normalize_observations: true
num_envs: 8192
num_evals: 10
num_minibatches: 32
num_resets_per_eval: 1
num_timesteps: 200_000_000
num_updates_per_batch: 4
reward_scaling: 1.0
unroll_length: 20
