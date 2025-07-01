import click
import wandb
from src import dqn

@click.group()
def cli():
    pass

@cli.command()
@click.option('--use-wandb', is_flag=True, help='Enable wandb (default: disable)')
def run_training_dqn(use_wandb: bool):
    try:
        if use_wandb:
            wandb.init(project="dqn", mode="online")
        else:
            wandb.init(project="dqn", mode="disabled")
        dqn.main(env_id="Breakout-v4", outdir="out/dqn")
    finally:
        wandb.finish()


if __name__ == "__main__":
    cli()



