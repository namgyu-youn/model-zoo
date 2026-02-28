"""3-bit: GPTQ + EoRA with mse=2.0 + Enhanced Smoothing

KEY OPTIMIZATIONS:
1. mse=2.0: Grid search (vs 0.1)
2. group_size=64
3. SmoothMSE: steps=64, maxshrink=0.70 (enhanced)
4. 512 samples: Better coverage

Checkpoint: https://huggingface.co/namgyu-youn/EXAONE-4.0-1.2B-GPTQ-W3A16-EoRA

Perplexity: <TBU>

Throughput: <TBU>
"""
import torch
import shutil
from pathlib import Path
from gptqmodel import GPTQModel
from gptqmodel.quantization import (
    QuantizeConfig, FORMAT, METHOD, HessianConfig,
    FailSafe, FailSafeStrategy, SmoothMSE
)
from gptqmodel.quantization.config import VramStrategy
from gptqmodel.adapter.adapter import Lora
from scripts.utils import prepare_calibration_data, print_section, upload_to_hf


BASE_MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"
HF_REPO_QUANT = "namgyu-youn/EXAONE-4.0-1.2B-GPTQ-W3A16"
HF_REPO_EORA = "namgyu-youn/EXAONE-4.0-1.2B-GPTQ-W3A16-EoRA"
BITS = 3
GROUP_SIZE = 128
NUM_SAMPLES = 512
EORA_RANK = 96  # Higher for 3-bit


def main():
    print_section("3-bit: mse=2.0 + Enhanced Config")

    calibration_data = prepare_calibration_data(num_samples=NUM_SAMPLES)

    quantize_config = QuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        sym=True,
        desc_act=False,
        act_group_aware=True,

        mse=2.0,  # Lower than 4-bit for 3-bit stability

        damp_percent=0.06,  # Higher for 3-bit
        damp_auto_increment=0.01,

        failsafe=FailSafe(
            strategy=FailSafeStrategy.MEDIAN,
            threshold="0.5%",
            smooth=SmoothMSE(
                steps=64,
                maxshrink=0.70,  # More aggressive for 3-bit
                group_size_threshold=GROUP_SIZE
            )
        ),

        hessian=HessianConfig(
            chunk_size=None,
            chunk_bytes=512*1024*1024,
            staging_dtype=torch.float16
        ),

        offload_to_disk=False,
        vram_strategy=VramStrategy.EXCLUSIVE,
    )

    model = GPTQModel.from_pretrained(
        BASE_MODEL,
        quantize_config=quantize_config,
        trust_remote_code=True,
        dtype=torch.float16
    )

    print("Quantizing (60-90 min)...")
    model.quantize(calibration_data, batch_size=1)

    temp_quant_path = f"{HF_REPO_EORA}_temp"
    model.save(temp_quant_path)

    # Upload quantized model (without EoRA)
    card_quant = f"""---
tags: [quantization, gptq, 3-bit]
base_model: {BASE_MODEL}
---

# EXAONE-4.0-1.2B GPTQ W3

**3-bit**: mse=2.0 + group_size={GROUP_SIZE} + SmoothMSE(64,0.70)

## Usage

### GPTQModel
```python
from gptqmodel import GPTQModel
model = GPTQModel.from_quantized("{HF_REPO_QUANT}", device="cuda:0")
```

### vLLM
```python
from vllm import LLM
llm = LLM(model="{HF_REPO_QUANT}", dtype="float16")
```
"""
    upload_to_hf(model, HF_REPO_QUANT, card_quant)

    eora_adapter = Lora(
        path=f"{temp_quant_path}/eora_rank{EORA_RANK}",
        rank=EORA_RANK,
    )

    print(f"Generating EoRA rank={EORA_RANK} (20-40 min)...")
    GPTQModel.adapter.generate(
        adapter=eora_adapter,
        model_id_or_path=BASE_MODEL,
        quantized_model_id_or_path=temp_quant_path,
        calibration_dataset=calibration_data,
        calibration_dataset_concat_size=0,
    )

    # Upload quantized model with EoRA
    card_eora = f"""---
tags: [quantization, gptq, 3-bit, eora]
base_model: {BASE_MODEL}
---

# EXAONE-4.0-1.2B GPTQ W3 + EoRA

**3-bit**: mse=2.0 + group_size={GROUP_SIZE} + SmoothMSE(64,0.70) + EoRA(rank={EORA_RANK})

Expected: 94-96% quality × 3.5-5.0x = **3.29-4.80 score**

## Usage

### GPTQModel
```python
from gptqmodel import GPTQModel
model = GPTQModel.from_quantized("{HF_REPO_EORA}", device="cuda:0")
```

### vLLM
```python
from vllm import LLM
llm = LLM(model="{HF_REPO_EORA}", dtype="float16")
```
"""
    upload_to_hf(None, HF_REPO_EORA, card_eora, upload_dir=temp_quant_path)

    shutil.rmtree(Path(temp_quant_path))


if __name__ == "__main__":
    main()
