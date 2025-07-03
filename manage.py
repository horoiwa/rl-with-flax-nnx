import click
import wandb
from src import dqn

@click.group()
def cli():
    pass

@cli.command(name="dqn")
@click.option('--use-wandb', is_flag=True, help='Enable wandb (default: disable)')
def run_dqn(use_wandb: bool):
    try:
        wandb.init(
            project="dqn",
            mode="online" if use_wandb else "disabled",
        )
        dqn.main(env_id="Breakout-v4", outdir="out/dqn")
    finally:
        wandb.finish()


if __name__ == "__main__":
    cli()



