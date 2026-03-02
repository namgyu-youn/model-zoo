"""4-bit: GPTQ + EoRA with mse=2.4 + Enhanced Smoothing

KEY OPTIMIZATIONS:
1. mse=2.4: Grid search for optimal scales/zeros (MAJOR boost)
2. group_size=128
3. SmoothMSE: steps=64, maxshrink=0.75 (vs steps=16, maxshrink=0.9)
4. 512 samples: Better coverage

Checkpoint: https://huggingface.co/namgyu-youn/EXAONE-4.0-1.2B-GPTQ-W4A16-EoRA

Perplexity:
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6621|±  |0.0209|
|     |       |strict-match    |     5|exact_match|↑  |0.6562|±  |0.0210|

Throughput: 10.36 requests/s, 11940.46 total tokens/s, 1326.72 output tokens/s

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
HF_REPO_QUANT = "namgyu-youn/EXAONE-4.0-1.2B-GPTQ-W4A16"
HF_REPO_EORA = "namgyu-youn/EXAONE-4.0-1.2B-GPTQ-W4A16-EoRA"
BITS = 4
GROUP_SIZE = 128
NUM_SAMPLES = 512  # Calibration samples
EORA_RANK = 64  # Error-Optimized Rank Adaptation: higher = better recovery


def main():
    print_section("4-bit: mse=2.4 + Enhanced Config")

    # Calibration
    calibration_data = prepare_calibration_data(num_samples=NUM_SAMPLES)

    quantize_config = QuantizeConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        sym=True,
        desc_act=False,
        act_group_aware=True,

        mse=2.4,  # Grid search optimization

        # Hessian dampening (regularization)
        damp_percent=0.01,
        damp_auto_increment=0.005,
        # Failsafe: triggers smoothing when quantization error exceeds threshold
        failsafe=FailSafe(
            strategy=FailSafeStrategy.MEDIAN,
            threshold="0.5%",  # Trigger if layer error > 0.5%
            smooth=SmoothMSE(
                steps=64,  # Search iterations
                maxshrink=0.75,  # Max shrink factor
                group_size_threshold=GROUP_SIZE
            )
        ),

        # Hessian computation: controls memory usage during quantization
        hessian=HessianConfig(
            chunk_size=None,
            chunk_bytes=512*1024*1024,
            staging_dtype=torch.float16
        ),

        offload_to_disk=False,
        vram_strategy=VramStrategy.EXCLUSIVE,  # Optimal for non-MoE models
    )

    # Quantize
    model = GPTQModel.from_pretrained(
        BASE_MODEL,
        quantize_config=quantize_config,
        trust_remote_code=True,
        dtype=torch.float16
    )

    print("Quantizing (40-70 min)...")
    model.quantize(calibration_data, batch_size=1)

    # Save temp for EoRA
    temp_quant_path = f"{HF_REPO_EORA}_temp"
    model.save(temp_quant_path)

    # Upload quantized model (without EoRA)
    card_quant = f"""---
tags: [quantization, gptq, 4-bit]
base_model: {BASE_MODEL}
---

# EXAONE-4.0-1.2B GPTQ W4

**4-bit**: mse=2.4 + group_size={GROUP_SIZE} + SmoothMSE(64,0.75)

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

    # EoRA
    eora_adapter = Lora(
        path=f"{temp_quant_path}/eora_rank{EORA_RANK}",
        rank=EORA_RANK,
    )

    print("Generating EoRA (15-30 min)...")
    GPTQModel.adapter.generate(
        adapter=eora_adapter,
        model_id_or_path=BASE_MODEL,
        quantized_model_id_or_path=temp_quant_path,
        calibration_dataset=calibration_data,
        calibration_dataset_concat_size=0,
    )

    # Upload quantized model with EoRA
    card_eora = f"""---
tags: [quantization, gptq, 4-bit, eora]
base_model: {BASE_MODEL}
---

# EXAONE-4.0-1.2B GPTQ W4 + EoRA

**4-bit**: mse=2.4 + group_size={GROUP_SIZE} + SmoothMSE(64,0.75) + EoRA(rank={EORA_RANK})

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

    # Cleanup
    shutil.rmtree(Path(temp_quant_path))


if __name__ == "__main__":
    main()
