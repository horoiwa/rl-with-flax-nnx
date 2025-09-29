from pathlib import Path
import click
import wandb


def train(env_id: str, log_dir: str):
    pass


def evaluate(env_id: str, log_dir: str, n_episodes: int, record_video: bool, seed: int):
    pass


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
            project="grpo",
            mode="online" if use_wandb else "disabled",
        )
        train(env_id=env_id, log_dir=f"{log_dir}/{env_id}")
    finally:
        wandb.finish()


@cli.command(name="eval")
@click.option("--env-id", default="Go1JoystickFlatTerrain", help="Environment ID")
@click.option("--log-dir", default="log", help="Directory to save logs and videos")
@click.option("--seed", default=0, help="seed")
def run_evaluation(env_id: str, log_dir: str, seed: int):
    evaluate(
        env_id=env_id,
        log_dir=f"{log_dir}/{env_id}",
        n_episodes=5,
        record_video=True,
        seed=seed,
    )


if __name__ == "__main__":
    cli()
