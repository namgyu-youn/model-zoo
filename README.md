# model-zoo

This project aims to experiment eager (model) level optimization. It includes: (1) quantization, (2) triton kernel, (3) GPU profiling, and more.

## Quick Start

### Setup

```bash
bash install.sh
```

### Run GPTQModel

```bash
python scripts/gptqmodel/4bit.py # W4A16-INT
python scripts/gptqmodel/3bit.py # W3A16-INT
python scripts/gptqmodel/mixed_4_3bit.py # Mixed precision (Attention to 4-bit, MLP to 3-bit)
```

### Run Benchmark

```bash
export MODEL=<MODEL>

# Accuracy (Perplexity)
lm_eval --model vllm \
    --model_args pretrained=<MODEL>,dtype=float16,gpu_memory_utilization=0.85,enable_thinking=False,max_gen_toks=2048 \
    --tasks gsm8k \
    --limit 512 \
    --output_path results \
    --apply_chat_template \
    --batch_size auto

# Throughput
vllm bench throughput \
  --input-len 256 \
  --output-len 256 \
  --model <MODEL> \
  --num-prompts 100 \
  --max-model-len 4096 \
  --enforce-eager
```
