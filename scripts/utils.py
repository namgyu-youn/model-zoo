from typing import Optional
import tempfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, get_token


def prepare_calibration_data(tokenizer, num_samples: int = 512):
    """Load MANTA-1M and format with chat template."""
    ds = load_dataset("LGAI-EXAONE/MANTA-1M", split=f"train[:{num_samples}]")

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=False,
                tokenize=False,
            )
        }

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=512,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(preprocess)
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    return ds


def print_section(title: str) -> None:
    """Print formatted section header"""
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def upload_to_hf(
    model,
    repo_id: str,
    model_card: str,
    upload_dir: Optional[str] = None,
) -> None:
    """Upload quantized model to HF Hub"""
    token = get_token()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = upload_dir or tmpdir
        if not upload_dir:
            print("📦 Saving model...")
            model.save_quantized(tmpdir)

        Path(path, "README.md").write_text(model_card)

        create_repo(repo_id, token=token, exist_ok=True)
        print(f"☁️ Uploading to {repo_id}...")
        HfApi(token=token).upload_folder(
            folder_path=path, repo_id=repo_id, token=token
        )
        print(f"✅ https://huggingface.co/{repo_id}")
