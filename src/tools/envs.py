from typing import Optional

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, Timelimit, RecordVideo


def get_atari_env(
    env_id: str,
    video_folder: Optional[Path] = None,
) -> gym.Env:
    """Create and return an Atari environment with preprocessing."""
    _env = gym.make(
        env_id,
        render_mode=render_mode,
        frameskip=1,
        repeat_action_probability=0.0,
    )
    if video_folder is not None:
        _env = RecordVideo(
            env=_env,
            video_folder=video_folder,
            episode_trigger=lambda x: True,
        )

    env = Timelimit(
        AtariPreprocessing(
            env=_env,
            noop_max=10,
            frame_skip=4,
            terminal_on_life_loss=True,
            screen_size=84,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=True,
        ),
        max_episode_steps=2000,
    )
    try:
        yield env
    finally:
        env.close()
