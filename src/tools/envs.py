from typing import Optional
from pathlib import Path

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TimeLimit, FrameStack, RecordVideo


ENV_INFO = {
    "Breakout": {"env_name": "ALE/Breakout-v5", "action_space": 4},
    "CartPole": {"env_name": "CartPole-v1", "action_space": 2},
}


def get_atari_env(
    env_id: str,
    render_mode: str = "rgb_array",
    video_folder: Optional[Path] = None,
) -> gym.Env:
    """Create and return an Atari environment with preprocessing."""
    env_name = ENV_INFO[env_id]["env_name"]
    env = TimeLimit(
        FrameStack(
            AtariPreprocessing(
                env=gym.make(
                    env_name,
                    render_mode=render_mode,
                    frameskip=1,
                    repeat_action_probability=0.0,
                ),
                noop_max=10,
                frame_skip=4,
                terminal_on_life_loss=True,
                screen_size=84,
                grayscale_obs=True,
                grayscale_newaxis=False,
                scale_obs=True,
            ),
            num_stack=4,
        ),
        max_episode_steps=2000,
    )

    if video_folder is not None:
        env = RecordVideo(
            env=env,
            video_folder=video_folder,
            episode_trigger=lambda x: x % 25 == 0,
        )

    return env


def get_action_space(env_id: str) -> int:
    """Return the action space size for the given environment."""
    return gym.make(env_id).action_space.n
