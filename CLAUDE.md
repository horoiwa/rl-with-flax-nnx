# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a collection of deep reinforcement learning algorithm implementations using Flax NNX. The project focuses on educational clarity over code reuse - each algorithm implementation is designed to be as independent as possible rather than sharing common components. Many hyperparameters are hardcoded for simplicity.

**Languages**: Python (JAX/Flax ecosystem)
**Validation Environment**: ALE/Breakout (Atari games)

## Architecture

### Core Structure
- `src/` - Algorithm implementations and utilities
  - `dqn.py` - Deep Q-Network implementation (partially complete)
  - `a2c.py` - Advantage Actor-Critic (empty/template)
  - `ppo.py` - Proximal Policy Optimization (empty/template)
  - `tools/` - Shared utilities
    - `envs.py` - Environment setup and preprocessing for Atari games
    - `buffers.py` - Replay buffer implementations (empty/template)

### Key Dependencies
- **Flax NNX**: Neural network framework (v0.10.6+)
- **JAX**: Numerical computing with GPU support
- **Gymnasium**: RL environment interface with Atari and MuJoCo support
- **Ray**: Distributed computing framework

### Environment Setup
The project uses Atari environments with specific preprocessing:
- AtariPreprocessing wrapper with 84x84 grayscale images
- Frame skipping (4 frames), action repeat prevention
- Episode time limits (2000 steps)
- Life loss treated as terminal state

## Development Commands

### Installation
```bash
# Install dependencies
uv sync
```

### Running Algorithms
```bash
# Train specific algorithm (when train.sh is implemented)
./train.sh --algo dqn --env breakout
./train.sh --algo a2c --env breakout  
./train.sh --algo ppo --env breakout
```

### Direct Execution
```bash
# Run DQN directly
python -m src.dqn
```

## Implementation Notes

- Each algorithm follows a similar structure: Network class, Buffer class, training functions
- DQN implementation includes: DQNCNN network, ReplayBuffer, epsilon scheduling, network syncing
- Environments use context manager pattern (`get_atari_env` yields environment)
- No test suite currently implemented
- No linting/formatting tools configured in pyproject.toml

## Known Issues
- `ENV_INFO` is imported but not defined in `src/tools/envs.py`
- Several template files are empty (a2c.py, ppo.py, buffers.py)
- Missing implementation details in DQN class methods (marked with `...`)
- `train.sh` script exists but is empty