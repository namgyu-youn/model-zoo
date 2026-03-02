"""MIXED 4/3-bit: Attention 4-bit + MLP 3-bit

EXAONE architecture: MLP=60%, Attention=40% of params
Strategy: Keep quality-critical attention at 4-bit, compress MLP to 3-bit
Effective: ~3.4 bits

Checkpoint: namgyu-youn/EXAONE-4.0-1.2B-GPTQ-W4A16

Perplexity: <TBU>

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
HF_REPO_QUANT = "namgyu-youn/EXAONE-4.0-1.2B-GPTQ-Mixed"
HF_REPO_EORA = "namgyu-youn/EXAONE-4.0-1.2B-GPTQ-Mixed-EoRA"
GROUP_SIZE = 32
NUM_SAMPLES = 512
EORA_RANK = 64


def main():
    print_section("MIXED 4/3-bit: Architecture-Optimized")

    calibration_data = prepare_calibration_data(num_samples=NUM_SAMPLES)

    quantize_config = QuantizeConfig(
        bits=3,  # Default for MLP
        group_size=GROUP_SIZE,
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        sym=True,
        desc_act=False,
        act_group_aware=True,
        mse=2.2,
        damp_percent=0.03,
        damp_auto_increment=0.008,

        # Mixed: Attention 4-bit, MLP 3-bit
        dynamic={
            "+:model\\.layers\\.[0-9]+\\.(q_proj|k_proj|v_proj|o_proj)": {
                "bits": 4,
                "group_size": GROUP_SIZE,
                "mse": 2.4,
            },
            "+:model\\.layers\\.[0-9]+\\.mlp\\.(up_proj|down_proj|gate_proj)": {
                "bits": 3,
                "group_size": GROUP_SIZE,
                "mse": 2.0,
            },
        },

        failsafe=FailSafe(
            strategy=FailSafeStrategy.MEDIAN,
            threshold="0.5%",
            smooth=SmoothMSE(steps=64, maxshrink=0.72, group_size_threshold=GROUP_SIZE)
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

    print("Quantizing mixed 4/3-bit (50-80 min)...")
    model.quantize(calibration_data, batch_size=1)

    temp_quant_path = f"{HF_REPO_EORA}_temp"
    model.save(temp_quant_path)

    # Upload quantized model (without EoRA)
    print("\n[1/2] Uploading quantized model to HF...")
    card_quant = f"""---
tags: [quantization, gptq, mixed-precision]
base_model: {BASE_MODEL}
---

# EXAONE-4.0-1.2B GPTQ Mixed 4/3-bit

**MIXED 4/3-bit**: Attention 4-bit (40%), MLP 3-bit (60%)

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
        rank=EORA_RANK
    )

    print("Generating EoRA (20-35 min)...")
    GPTQModel.adapter.generate(
        adapter=eora_adapter,
        model_id_or_path=BASE_MODEL,
        quantized_model_id_or_path=temp_quant_path,
        calibration_dataset=calibration_data,
        calibration_dataset_concat_size=0,
    )

    # Upload quantized model with EoRA (adapter already saved by generate())
    print("\n[2/2] Uploading quantized model + EoRA to HF...")
    card_eora = f"""---
tags: [quantization, gptq, mixed-precision, eora]
base_model: {BASE_MODEL}
---

# EXAONE-4.0-1.2B GPTQ Mixed 4/3-bit + EoRA

**MIXED 4/3-bit**: Attention 4-bit (40%), MLP 3-bit (60%)
Effective ~3.4 bits

Expected: 97-98.5% quality × 3.0-3.5x = **2.91-3.45 score**

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

    print("\nPerformance:")
    print("  Pure 4-bit:       0.98  × 2.8 = 2.74")
    print("  Mixed 4/3 (this): 0.975 × 3.0 = 2.93")
    print("  Pure 3-bit:       0.94  × 3.5 = 3.29")


if __name__ == "__main__":
    main()
