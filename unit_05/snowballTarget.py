"""
Unit 05 - SnowballTarget (ML-Agents PPO)

Just press Run. The script will:
  1. Ask for the path to your SnowballTarget Unity executable
  2. Run mlagents-learn training as a subprocess
  3. Save the trained .onnx model locally
  4. Push the model to HuggingFace Hub

Requirements:
  pip install mlagents huggingface_hub
"""

import subprocess
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, login
from huggingface_hub.repocard import metadata_eval_result, metadata_save

# ── Config ─────────────────────────────────────────────────────────────────────
USERNAME      = "TheBestMoldyCheese"
ENV_ID        = "SnowballTarget-v0"
BEHAVIOR_NAME = "SnowballTarget"
RUN_ID        = "SnowballTarget"
CONFIG_PATH   = Path("unit_05/ml-agents/config/ppo/SnowballTarget.yaml")
RESULTS_DIR   = Path("results") / RUN_ID / BEHAVIOR_NAME
ONNX_MODEL    = RESULTS_DIR / f"{BEHAVIOR_NAME}.onnx"
SAVE_DIR      = Path("unit_05/saved_models")
REPO_ID       = f"{USERNAME}/ppo-{BEHAVIOR_NAME}"


# ── Training ───────────────────────────────────────────────────────────────────
def train(env_path: str):
    """Launch mlagents-learn and block until training is complete."""
    cmd = [
        "mlagents-learn",
        str(CONFIG_PATH),
        f"--env={env_path}",
        f"--run-id={RUN_ID}",
        "--no-graphics",
        "--force",   # overwrite previous run with same id
    ]
    print("\n--- Starting Training ---")
    print("Command:", " ".join(cmd))
    print("This will take a while. Watch the logs below...\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(
            f"mlagents-learn exited with code {result.returncode}. "
            "Check the logs above for errors."
        )
    print("\n--- Training Complete ---")


# ── Save ───────────────────────────────────────────────────────────────────────
def save_model(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    print(f"Model saved to {dest}")
    return dest


# ── Push to Hub ────────────────────────────────────────────────────────────────
def push_to_hub(onnx_path: Path):
    _, repo_name = REPO_ID.split("/")
    api = HfApi()
    repo_url = api.create_repo(repo_id=REPO_ID, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        local_dir = Path(tmp)

        shutil.copy2(onnx_path, local_dir / onnx_path.name)
        shutil.copy2(CONFIG_PATH, local_dir / CONFIG_PATH.name)

        metadata = {
            "tags": [
                ENV_ID,
                "deep-reinforcement-learning",
                "reinforcement-learning",
                "ML-Agents",
                "deep-rl-class",
            ]
        }

        eval_result = metadata_eval_result(
            model_pretty_name=repo_name,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value="N/A",
            dataset_pretty_name=ENV_ID,
            dataset_id=ENV_ID,
        )
        metadata = {**metadata, **eval_result}

        model_card = f"""# **PPO** Agent playing **{ENV_ID}**

This is a trained model of a **PPO** agent playing **{ENV_ID}** using the
[ML-Agents](https://github.com/Unity-Technologies/ml-agents) library.

To learn how to train your own agent check Unit 5 of the Deep Reinforcement
Learning Course:
https://huggingface.co/deep-rl-course/unit5/introduction

## Training details
- Trainer: PPO
- Config: `{CONFIG_PATH.name}`
- Run ID: `{RUN_ID}`
- Max steps: 1,000,000

## Usage
Load the `.onnx` model in the Unity SnowballTarget environment via ML-Agents inference mode.
"""

        readme_path = local_dir / "README.md"
        readme_path.write_text(model_card, encoding="utf-8")
        metadata_save(readme_path, metadata)

        api.upload_folder(
            repo_id=REPO_ID,
            folder_path=local_dir,
            path_in_repo=".",
        )

    print(f"\nModel pushed to Hub: {repo_url}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: get Unity exe path
    env_path = input(
        "Enter the path to your SnowballTarget Unity executable\n"
        "(e.g. C:/builds/SnowballTarget/SnowballTarget.exe): "
    ).strip()

    if not Path(env_path).exists():
        print(f"[ERROR] File not found: {env_path}")
        exit(1)

    # Step 2: train
    train(env_path)

    # Step 3: verify model was produced
    if not ONNX_MODEL.exists():
        print(f"[ERROR] Expected model not found at {ONNX_MODEL}")
        print("Training may have failed — check logs above.")
        exit(1)

    # Step 4: save locally
    save_model(ONNX_MODEL, SAVE_DIR)

    # Step 5: push to Hub
    print("\nLogging in to Hugging Face...")
    login()
    push_to_hub(ONNX_MODEL)
