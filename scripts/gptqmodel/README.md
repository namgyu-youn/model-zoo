# GPTQModel

This module quantizes [EXAONE-4.0](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B) and runs benchmarks inside [vLLM](https://github.com/vllm-project/vllm) and [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness), using [GPTQModel](https://github.com/ModelCloud/GPTQModel).


## 01. Benchmark Summary

| Bits | Perplexity (g8smk) | Throughput (tok/s) |
|------|-------------|-------------------|
| origin |        |         |
| 2bit   | 0.0000 |   TBU   |
| 3bit   |   TBU  |   TBU   |
| 4bit   | 0.6562 | 11,940.46 |

### How to run Benchmark

```bash
export MODEL=<MODEL>

# Accuracy (Perplexity)
lm_eval --model vllm \
    --model_args pretrained=<MODEL>,dtype=float16,gpu_memory_utilization=0.85,enable_thinking=False,max_gen_toks=2048 \
    --tasks gsm8k \
    --limit 512 \
    --output_path results \
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


## 02. Acknowledgement and Citation

```json
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

@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}

@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {The Language Model Evaluation Harness},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}
```