"""2-bit: GPTQ + EoRA with mse=1.5 + Aggressive Smoothing

KEY OPTIMIZATIONS:
1. mse=1.5: Lower for 2-bit stability (vs 0.15)
2. group_size=32: Maximum granularity
3. SmoothMSE: steps=96, maxshrink=0.65 (most aggressive)
4. 1024 samples: Maximum coverage for 2-bit
5. Skip embed_tokens + lm_head: Preserve critical I/O layers (34% of model)

Checkpoint: namgyu-youn/EXAONE-4.0-1.2B-GPTQ-W2A16

Perplexity (BROKEN):
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.0039|±  |0.0028|
|     |       |strict-match    |     5|exact_match|↑  |0.0000|±  |0.0000|

Throughput: <TBU>

Expected Score (acc x speed): <TBU>

Citations:
    @misc{qubitium2024gptqmodel,
      author = {ModelCloud.ai and qubitium@modelcloud.ai},
      title = {GPT-QModel},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {https://github.com/modelcloud/gptqmodel},
      note = {Contact: qubitium@modelcloud.ai},
      year = {2024},
    }

    @article{frantar-gptq,
      title={{GPTQ}: Accurate Post-training Compression for Generative Pretrained Transformers},
      author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
      journal={arXiv preprint arXiv:2210.17323},
      year={2022}
    }

    @article{liu2024eora,
      title={EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation},
      author={Liu, Shih-Yang and Yang, Huck and Wang, Chien-Yi and Fung, Nai Chit and Yin, Hongxu and Sakr, Charbel and Muralidharan, Saurav and Cheng, Kwang-Ting and Kautz, Jan and Wang, Yu-Chiang Frank and others},
      journal={arXiv preprint arXiv:2410.21271},
      year={2024}
    }

    @article{gar,
      title={Dual Precision Quantization for Efficient and Accurate Deep Neural Networks Inference, CVPRW 2025.},
      author={T. Gafni, A. Karnieli, Y. Hanani},
      journal={arXiv preprint arXiv:2505.14638},
      year={2025}
    }

    @article{frantar2024marlin,
      title={MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models},
      author={Frantar, Elias and Castro, Roberto L and Chen, Jiale and Hoefler, Torsten and Alistarh, Dan},
      journal={arXiv preprint arXiv:2408.11743},
      year={2024}
    }
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
HF_REPO_QUANT = "namgyu-youn/EXAONE-4.0-1.2B-GPTQ-W2A16"
HF_REPO_EORA = "namgyu-youn/EXAONE-4.0-1.2B-GPTQ-W2A16-EoRA"
BITS = 2
GROUP_SIZE = 32
NUM_SAMPLES = 1024  # Maximum for 2-bit
EORA_RANK = 128  # Highest for 2-bit recovery


def main():
    print_section("2-bit: mse=1.5 + Aggressive Config (EXPERIMENTAL)")

    calibration_data = prepare_calibration_data(num_samples=NUM_SAMPLES)

    quantize_config = QuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        sym=True,
        desc_act=False,
        act_group_aware=True,

        mse=1.5,  # Lower for 2-bit stability

        damp_percent=0.08,  # Highest for 2-bit
        damp_auto_increment=0.02,

        # Skip critical I/O layers: embed_tokens + lm_head (34% of model)
        dynamic={
            "-:model\\.embed_tokens": {},  # Skip input embedding (17%)
            "-:lm_head": {},                # Skip output head (17%)
        },

        failsafe=FailSafe(
            strategy=FailSafeStrategy.MEDIAN,
            threshold="1.0%",
            smooth=SmoothMSE(
                steps=96,  # Maximum search
                maxshrink=0.65,  # Most aggressive
                group_size_threshold=GROUP_SIZE
            )
        ),

        hessian=HessianConfig(
            chunk_size=None,
            chunk_bytes=512*1024*1024,
            staging_dtype=torch.float16
        ),

        offload_to_disk=False,
        vram_strategy=VramStrategy.EXCLUSIVE,  # Optimal for non-MoE models
    )

    model = GPTQModel.from_pretrained(
        BASE_MODEL,
        quantize_config=quantize_config,
        trust_remote_code=True,
        dtype=torch.float16
    )

    print("Quantizing (90-150 min)...")
    model.quantize(calibration_data, batch_size=1)

    temp_quant_path = f"{HF_REPO_EORA}_temp"
    model.save(temp_quant_path)

    # Upload quantized model (without EoRA)
    card_quant = f"""---
tags: [quantization, gptq, 2-bit]
base_model: {BASE_MODEL}
---

# EXAONE-4.0-1.2B GPTQ W2

**2-bit (EXPERIMENTAL)**: mse=1.5 + group_size={GROUP_SIZE} + SmoothMSE(96,0.65)
Skip embed_tokens + lm_head (34% preserved in FP16)

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

    print(f"Generating EoRA rank={EORA_RANK} (30-60 min)...")
    GPTQModel.adapter.generate(
        adapter=eora_adapter,
        model_id_or_path=BASE_MODEL,
        quantized_model_id_or_path=temp_quant_path,
        calibration_dataset=calibration_data,
        calibration_dataset_concat_size=0,
    )

    # Upload quantized model with EoRA
    card_eora = f"""---
tags: [quantization, gptq, 2-bit, eora]
base_model: {BASE_MODEL}
---

# EXAONE-4.0-1.2B GPTQ W2 + EoRA

**2-bit (EXPERIMENTAL)**: mse=1.5 + group_size={GROUP_SIZE} + SmoothMSE(96,0.65) + EoRA(rank={EORA_RANK})
Skip embed_tokens + lm_head (34% preserved in FP16)

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
