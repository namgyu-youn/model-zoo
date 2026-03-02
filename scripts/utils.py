"""Shared utility functions for quantization workflow"""
from typing import Optional
import os
import json
import tempfile
import subprocess
from pathlib import Path

from datasets import load_dataset, Dataset
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


def setup_gpu() -> str:
    """Setup TVM target for BitBLAS"""
    if target := os.environ.get("TVM_TARGET"):
        return target
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    gpu = result.stdout.lower()
    target = (
        "nvidia/nvidia-a100" if "a100" in gpu else
        "nvidia/nvidia-l4" if "l4" in gpu else
        "cuda"
    )
    os.environ["TVM_TARGET"] = target

    return target


# NOTE: Do we really need this?
def fix_chat_template(output_dir: str) -> None:
    """Fix EXAONE chat template and add it to tokenizer_config.json.

    1. Removes empty <think> tags from chat_template.jinja (if present)
    2. Adds chat_template field to tokenizer_config.json for lm_eval compatibility
    """
    path = Path(output_dir).resolve()
    if not path.is_dir():
        return

    # Fix chat_template.jinja
    template = path / "chat_template.jinja"
    if template.exists():
        content = template.read_text()
        fixes = [
            ('{{- "<think>\\n\\n</think>\\n\\n" }}', ''),
            (
                '{%- else %}\n                '
                '{{- "<think>\\n\\n</think>\\n\\n" }}\n'
                '            {%- endif %}',
                '{%- endif %}'
            ),
        ]
        for old, new in fixes:
            content = content.replace(old, new)
        template.write_text(content)

    # Add chat_template to tokenizer_config.json
    config_path = path / "tokenizer_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if 'chat_template' not in config:
            config['chat_template'] = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "[|system|]{{ message['content'] }}[|endofturn|]\\n"
                "{% elif message['role'] == 'user' %}"
                "[|user|]{{ message['content'] }}[|endofturn|]\\n"
                "{% elif message['role'] == 'assistant' %}"
                "[|assistant|]{{ message['content'] }}[|endofturn|]\\n"
                "{% endif %}{% endfor %}"
                "{% if add_generation_prompt %}[|assistant|]{% endif %}"
            )
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False)
            )

    print("✅ Chat template fixed")


def upload_to_hf(
    model,
    repo_id: str,
    model_card: str,
    upload_dir: Optional[str] = None,
) -> None:
    """Upload quantized model to HF Hub (GPTQModel)"""
    token = get_token()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = upload_dir or tmpdir
        if not upload_dir:
            print("📦 Saving model...")
            model.save_quantized(tmpdir)

        fix_chat_template(path)
        Path(path, "README.md").write_text(model_card)

        create_repo(repo_id, token=token, exist_ok=True)
        print(f"☁️ Uploading to {repo_id}...")
        HfApi(token=token).upload_folder(
            folder_path=path, repo_id=repo_id, token=token
        )
        print(f"✅ https://huggingface.co/{repo_id}")


def prepare_llmc_calibration_data(
    tokenizer, num_samples=512, max_length=512
) -> Dataset:
    """Prepare calibration dataset for llm-compressor"""
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    samples = []
    for ex in ds:
        if len(samples) >= num_samples:
            break
        text = ex['text'].strip()
        if len(text) > 100 and not text.startswith('='):
            samples.append({"text": text})

    ds = Dataset.from_list(samples)
    ds = ds.map(
        lambda x: tokenizer(
            x["text"],
            padding=False,
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        ),
        remove_columns=ds.column_names,
    )
    print(f"[INFO] Prepared {len(ds)} tokenized calibration samples")
    return ds


def upload_to_hf_llmc(
    model, tokenizer, repo_id: str, model_card: str,
    quantization_format: Optional[str] = "pack-quantized",
) -> None:
    """Upload llm-compressor quantized model to HF Hub.

    :param quantization_format: compression format for saving. Use
        "pack-quantized" (default) for int4/int8 packed weights — required for
        vLLM serving. Pass None to let llm-compressor infer the format.
    """
    token = get_token()

    with tempfile.TemporaryDirectory() as tmpdir:
        print("📦 Saving model...")
        model.save_pretrained(
            tmpdir, save_compressed=True,
            quantization_format=quantization_format,
        )
        tokenizer.save_pretrained(tmpdir)

        Path(tmpdir, "README.md").write_text(model_card)

        create_repo(repo_id, token=token, exist_ok=True)
        print(f"☁️ Uploading to {repo_id}...")
        HfApi(token=token).upload_folder(
            folder_path=tmpdir, repo_id=repo_id, token=token
        )
        print(f"✅ https://huggingface.co/{repo_id}")
