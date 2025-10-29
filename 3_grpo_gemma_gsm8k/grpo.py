import sys
import subprocess
import os
import csv
import re
from pathlib import Path

import click
from tqdm import tqdm
import wandb
import kagglehub

from flax import nnx
import grain
import sentencepiece as spm
import optax
from orbax import checkpoint as ocp

# ==============================================================================
# Path settings
# ==============================================================================
HOME = Path(__file__).parent
CACHE_DIR = HOME / "__cache__"
GSM8K_DATASET = "thedevastator/grade-school-math-8k-q-a"
VARIANT_NAME = "gemma3-1b-it"
MODEL_PATH = "google/gemma-3/Flax/" + VARIANT_NAME + "/1"
CKPT_DIR = HOME / "__checkpoints__"
os.environ["KAGGLEHUB_CACHE"] = str(CACHE_DIR.resolve())

# ==============================================================================
# Answer format settings
# ==============================================================================
REASONING_START = "<reasoning_start>"
REASONING_END = "<reasoning_end>"
SOLUTION_START = "<solution_start>"
SOLUTION_END = "<solution_end>"
SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {REASONING_START} and \
{REASONING_END}. Then, provide the final answer (i.e., just one numerical \
value) between {SOLUTION_START} and {SOLUTION_END}."""
TEMPLATE = """user
{system_prompt}

{question}
model"""
MATCH_FORMAT = re.compile(
    rf"^[\s]{{0,}}"
    rf"{REASONING_START}.+?{REASONING_END}.*?"
    rf"{SOLUTION_START}(.+?){SOLUTION_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

MATCH_NUMBERS = re.compile(
    rf"{SOLUTION_START}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)

# ==============================================================================
# Training parameters
# ==============================================================================
N_EPOCHS = 1
TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1

# ==============================================================================
# Generation parameters
# ==============================================================================
GENERATION_STEPS = 300
TEMPERATURE = 0.7
TOP_P = 0.95


def load_dataset():

    def dataset_from_csv_path(csv_path: Path):
        data = []
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(
                    {
                        "question": row["question"],
                        "answer": row["answer"],
                    }
                )
        dataset = (
            grain.MapDataset.source(data)
            .shuffle(seed=42)
            .map(
                lambda x: {
                    "prompt": TEMPLATE.format(
                        system_prompt=SYSTEM_PROMPT,
                        question=x["question"],
                    ),
                    "question": x["question"],
                    "answer": (
                        x["answer"].split("####")[1].strip()
                        if "####" in x["answer"]
                        else None
                    ),
                }
            )
        )

        return dataset

    dataset_dir = Path(kagglehub.dataset_download(GSM8K_DATASET))
    train_dataset = (
        dataset_from_csv_path(dataset_dir / "main_train.csv")
        .batch(TRAIN_BATCH_SIZE)
        .repeat(N_EPOCHS)
    )
    test_dataset = (
        dataset_from_csv_path(dataset_dir / "main_test.csv")
        .batch(TEST_BATCH_SIZE)
        .repeat(N_EPOCHS)
    )
    return train_dataset, test_dataset


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
        kagglehub.model_download(str(MODEL_PATH))

    weights_dir = CACHE_DIR / "models" / MODEL_PATH
    ckpt_path: Path = weights_dir / VARIANT_NAME
    params = params_lib.load_and_format_params(str(ckpt_path))
    transformer = transformer_lib.Transformer.from_params(params)

    vocab_path: Path = weights_dir / "tokenizer.model"
    vocab = spm.SentencePieceProcessor()
    vocab.Load(str(vocab_path))

    sampler = sampler_lib.Sampler(
        transformer=transformer,
        vocab=vocab,
    )
    # test_sampler(sampler)
    return transformer, vocab, sampler


def save_weights(model: nnx.Module, dir_name: str):
    if not CKPT_DIR.exists():
        CKPT_DIR.mkdir()
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(str(CKPT_DIR / dir_name), state, force=True)
    checkpointer.wait_until_finished()


def reward_fn(
    prompts: list[str],
    completions: list[str],
    true_answers: list[str],
    **kwargs,
) -> float:
    format_scores, accuracy_scores = _reward_fn(
        prompts=prompts, completions=completions, true_answers=true_answers, **kwargs
    )
    return [s1 + s2 for s1, s2 in zip(format_scores, accuracy_scores, strict=True)]


def _reward_fn(
    prompts: list[str],
    completions: list[str],
    true_answers: list[str],
    **kwargs,
) -> float:
    # ==========================================================================
    # Format reward
    # ==========================================================================
    scores_match_format_exactly = [
        3.0 if MATCH_FORMAT.search(res) is not None else 0 for res in completions
    ]
    scores_match_format_approximately = [
        sum(
            [
                0.5 if res.count(REASONING_START) == 1 else -0.5,
                0.5 if res.count(REASONING_END) == 1 else -0.5,
                0.5 if res.count(SOLUTION_START) == 1 else -0.5,
                0.5 if res.count(SOLUTION_END) == 1 else -0.5,
            ]
        )
        for res in completions
    ]

    # ==========================================================================
    # Accuracy reward
    # ==========================================================================
    scores_answer = []
    for res, true_answer in zip(completions, true_answers):
        guess: str | None = (
            match.group(1) if (match := MATCH_FORMAT.search(res)) is not None else None
        )
        score = 0
        if guess is None:
            pass
        elif guess == true_answer:
            score += 3.0
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:
                    score += 0.5
                elif 0.8 <= ratio <= 1.2:
                    score += 0.25
                else:
                    score -= 1.0
            except:
                score -= 0.5
        scores_answer.append(score)

    scores_number = []
    for res, true_answer in zip(completions, true_answers):
        guess = (
            match.group(1) if (match := MATCH_NUMBERS.search(res)) is not None else None
        )
        score = 0
        if guess is None:
            pass
        else:
            try:
                true_answer = float(true_answer.strip())
                guess = float(guess.strip())
                score += 1.5 if guess == true_answer else 0.0
            except:
                pass
        scores_number.append(score)

    # ==========================================================================
    # Total reward
    # ==========================================================================
    scores_format = [
        s1 + s2
        for s1, s2 in zip(
            scores_match_format_exactly,
            scores_match_format_approximately,
            strict=True,
        )
    ]
    scores_accuracy = [
        s3 + s4
        for s3, s4 in zip(
            scores_answer,
            scores_number,
            strict=True,
        )
    ]
    return scores_format, scores_accuracy


def generate(
    sampler,
    prompts: list[str],
    generation_steps=GENERATION_STEPS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    seed=None,
):
    out_data = sampler(
        input_strings=prompts,
        total_generation_steps=generation_steps,
        temperature=temperature,
        top_p=top_p,
        echo=False,
        seed=seed,
    )
    return out_data.text


def evaluate(sampler, dataset):
    for batch in tqdm(dataset):
        prompts = batch["prompt"]
        true_answers = batch["answer"]
        responses = generate(sampler, prompts)
        scores_format, scores_accuracy = _reward_fn(
            prompts=prompts,
            completions=responses,
            true_answers=true_answers,
        )
        print(scores_format, scores_accuracy)


def main(env_id: str, log_dir: str):
    train_dataset, test_dataset = load_dataset()
    gemma, vocab, sampler = load_model()
    evaluate(sampler, test_dataset)


@click.group()
def cli():
    pass


@cli.command(name="train")
@click.option("--env-id", default="Go1JoystickFlatTerrain", help="Environment ID")
@click.option("--log-dir", default="log", help="Directory to save logs and videos")
@click.option("--use-wandb", is_flag=True, help="Enable wandb (default: disable)")
def _(env_id: str, log_dir: str, use_wandb: bool):
    try:
        wandb.init(
            project="grpo",
            mode="online" if use_wandb else "disabled",
        )
        main(env_id=env_id, log_dir=f"{log_dir}/{env_id}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    cli()
