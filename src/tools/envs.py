from typing import Optional
from pathlib import Path

import ale_py
import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    TimeLimit,
    FrameStackObservation,
    RecordVideo,
)
import numpy as np


class ChannelLastFrameStack(FrameStackObservation):
    """Frame stacking wrapper that outputs in channel-last format (H, W, C)."""

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Convert from (stack_size, H, W) to (H, W, stack_size)
        obs = np.transpose(obs, (1, 2, 0))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Convert from (stack_size, H, W) to (H, W, stack_size)
        obs = np.transpose(obs, (1, 2, 0))
        return obs, reward, terminated, truncated, info


def get_atari_env(
    env_id: str,
    render_mode: str = "rgb_array",
    video_folder: Optional[Path] = None,
) -> gym.Env:
    """Create and return an Atari environment with preprocessing."""
    env = gym.make(
        env_id,
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=0.0,
    )
    if video_folder is not None:
        env = RecordVideo(
            env=env,
            video_folder=video_folder,
            episode_trigger=lambda x: x % 50 == 0,
        )
    env = TimeLimit(
        ChannelLastFrameStack(
            AtariPreprocessing(
                env=env,
                noop_max=10,
                frame_skip=4,
                terminal_on_life_loss=True,
                screen_size=84,
                grayscale_obs=True,
                grayscale_newaxis=False,
                scale_obs=True,
            ),
            stack_size=4,
        ),
        max_episode_steps=2000,
    )

    try:
        yield env
    finally:
        env.close()
