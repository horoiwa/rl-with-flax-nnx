
## Setup

Install `uv` in advance:  https://docs.astral.sh/uv/getting-started/installation/

```
cd ./dqn_atari_breakout
uv sync
```

## Run training

`uv run python manage.py --use-wandb`

**NOTE:**
If you get the error of "opencv-python package not installed, run `pip install gym[other]` to get dependencies for atari", please install `libgl1`

`apt update && apt install libgl1`

## References

[Flax NNX](https://flax.readthedocs.io/en/v0.8.3/experimental/nnx/index.html)

[DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)

[Gymnasium](https://gymnasium.farama.org/)

[ALE](https://ale.farama.org/)

