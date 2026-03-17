"""
Unit 05 - SnowballTarget (ML-Agents PPO)
Train with:
    mlagents-learn config/ppo/SnowballTarget.yaml --env=<path_to_SnowballTarget_exe> --run-id=SnowballTarget --no-graphics
Then push to Hub by running this script after training.
"""

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, login
from huggingface_hub.repocard import metadata_eval_result, metadata_save

# ── Config ────────────────────────────────────────────────────────────────────
USERNAME = "TheBestMoldyCheese"
ENV_ID = "SnowballTarget-v0"
BEHAVIOR_NAME = "SnowballTarget"
RUN_ID = "SnowballTarget"
# Path where mlagents-learn saves the trained model
RESULTS_DIR = Path("results") / RUN_ID / BEHAVIOR_NAME
ONNX_MODEL = RESULTS_DIR / f"{BEHAVIOR_NAME}.onnx"
CONFIG_PATH = Path("unit_05/ml-agents/config/ppo/SnowballTarget.yaml")
REPO_ID = f"{USERNAME}/ppo-{BEHAVIOR_NAME}"

# ── Save helper ───────────────────────────────────────────────────────────────
def save_model(src_onnx: Path, dest_dir: Path):
    """Copy the trained .onnx model to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src_onnx.name
    shutil.copy2(src_onnx, dest)
    print(f"Model saved to {dest}")
    return dest


# ── Push to Hub ───────────────────────────────────────────────────────────────
def push_to_hub(
    repo_id: str,
    onnx_path: Path,
    config_path: Path,
    env_id: str,
    behavior_name: str,
):
    _, repo_name = repo_id.split("/")
    api = HfApi()

    # Create / ensure repo exists
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        local_dir = Path(tmp)

        # Copy model
        shutil.copy2(onnx_path, local_dir / onnx_path.name)

        # Copy config
        shutil.copy2(config_path, local_dir / config_path.name)

        # Metadata & model card
        metadata = {}
        metadata["tags"] = [
            env_id,
            "deep-reinforcement-learning",
            "reinforcement-learning",
            "ML-Agents",
            "deep-rl-class",
        ]

        eval_result = metadata_eval_result(
            model_pretty_name=repo_name,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value="N/A",          # fill in after you evaluate
            dataset_pretty_name=env_id,
            dataset_id=env_id,
        )
        metadata = {**metadata, **eval_result}

        model_card = f"""# **PPO** Agent playing **{env_id}**

This is a trained model of a **PPO** agent playing **{env_id}** using the
[ML-Agents](https://github.com/Unity-Technologies/ml-agents) library.

To learn how to train your own agent check Unit 5 of the Deep Reinforcement
Learning Course:
https://huggingface.co/deep-rl-course/unit5/introduction

## Training details
- Trainer: PPO
- Config: `{config_path.name}`
- Trained with: `mlagents-learn {config_path} --env=<SnowballTarget_exe> --run-id={behavior_name} --no-graphics`

## Usage
Load the `.onnx` model in the Unity SnowballTarget environment via the
ML-Agents inference mode.
"""

        readme_path = local_dir / "README.md"
        readme_path.write_text(model_card, encoding="utf-8")
        metadata_save(readme_path, metadata)

        # Upload
        api.upload_folder(
            repo_id=repo_id,
            folder_path=local_dir,
            path_in_repo=".",
        )

    print(f"Model pushed to Hub: {repo_url}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not ONNX_MODEL.exists():
        print(f"[ERROR] Trained model not found at: {ONNX_MODEL}")
        print("Train first with:")
        print(
            f"  mlagents-learn {CONFIG_PATH} "
            f"--env=<path_to_SnowballTarget_exe> --run-id={RUN_ID} --no-graphics"
        )
    else:
        # Save a local copy
        save_dir = Path("unit_05/saved_models")
        save_model(ONNX_MODEL, save_dir)

        # Push to Hub
        login()
        push_to_hub(
            repo_id=REPO_ID,
            onnx_path=ONNX_MODEL,
            config_path=CONFIG_PATH,
            env_id=ENV_ID,
            behavior_name=BEHAVIOR_NAME,
        )
