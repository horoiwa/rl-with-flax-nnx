import sys
import subprocess
import os

from pathlib import Path
import click
import wandb
import kagglehub

from flax import nnx
import sentencepiece as spm

HOME = Path(__file__).parent
CACHE_DIR = HOME / "__cache__"
VARIANT_NAME = "gemma3-1b-it"
MODEL_PATH = "google/gemma-3/Flax/" + VARIANT_NAME


def load_model():
    # Set a custom cache directory for KaggleHub
    # This must be done before importing/using kagglehub

    # Clone the Gemma repository if it doesn't exist
    if not (CACHE_DIR / "flax").exists():
        subprocess.run(
            ["git", "clone", "https://github.com/google/flax.git", f"{CACHE_DIR}/flax"]
        )

    # Import the necessary modules from the Gemma repository
    sys.path.append(f"{CACHE_DIR}/flax/examples/gemma")
    import params as params_lib
    import sampler as sampler_lib
    import transformer as transformer_lib

    # Clean up sys.path after import nessessary modules
    sys.path.pop()

    # Download the model weights and tokenizer if they don't exist
    if not (CACHE_DIR / "models" / MODEL_PATH).exists():
        print("Download model weights and tokenizer from KaggleHub")
        print("This requires Kaggle account and APIkey")
        print("You also need to agree to the Gemma3 model license on KaggleHub first:")
        kagglehub.login()
        os.environ["KAGGLEHUB_CACHE"] = str(CACHE_DIR.resolve())
        weights_dir: str = kagglehub.model_download(str(MODEL_PATH))

    ckpt_path: str = os.path.join(weights_dir, VARIANT_NAME)
    params = params_lib.load_and_format_params(ckpt_path)
    transformer = transformer_lib.Transformer.from_params(params)
    nnx.display(transformer)

    vocab_path: str = os.path.join(weights_dir, "tokenizer.model")
    vocab = spm.SentencePieceProcessor()
    vocab.Load(vocab_path)

    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
    )
    return transformer, vocab, sampler


def test_sampler(sampler):
    input_batch = [
        "\n# Python program for implementation of Bubble Sort\n\ndef bubbleSort(arr):",
    ]

    out_data = sampler(
        input_strings=input_batch,
        total_generation_steps=300,  # The number of steps performed when generating a response.
    )

    for input_string, out_string in zip(input_batch, out_data.text):
        print(f"Prompt:\n{input_string}\nOutput:\n{out_string}")
        print()
        print(10 * "#")


def train(env_id: str, log_dir: str):
    transformer, vocab, sampler = load_model()
    test_sampler(sampler)
    import pdb; pdb.set_trace()  # fmt: skip


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
