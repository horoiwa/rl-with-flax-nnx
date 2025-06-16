import jax
import jax.numpy as jnp
from flax import nnx

import ale_py
import gymnasium as gym


class Linear(nnx.Module):
    def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))
        self.din, self.dout = din, dout

    def __call__(self, x: jax.Array):
        return x @ self.w + self.b


def main():
    env = gym.make(
        "Breakout-v4",
        render_mode="rgb_array",
        frameskip=4,
        repeat_action_probability=0.0,
    )
    rngs = nnx.Rngs(params=jax.random.PRNGKey(42))
    model1 = Linear(4, 2, rngs=rngs)
    model2 = Linear(4, 2, rngs=rngs)
    x = jnp.ones((1, 4))
    y = model1(x)
    print("Model output:", y)


if __name__ == "__main__":
    main()
