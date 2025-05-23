# dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching

Official PyTorch implementation of the paper **["dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching"](./asset/paper.pdf)** (dLLM-Cache).

## :fire: News

- [2025/05/23] The code of our paper has been released.
- [2025/05/22] Our paper has been released.



## ✨️ Key Highlights
<!-- Our approach excels across diverse tasks, as shown in the radar chart below: -->
![radar_speed](./asset/radar.png)

<!-- Diffusion-based Large Language Models (dLLMs) offer a robust alternative to Autoregressive Models (ARMs) by iteratively denoising masked text segments. However, their bidirectional attention mechanism results in high inference latency, making traditional ARM acceleration methods like Key-Value caching incompatible.

**dLLM-Cache** is a **training-free adaptive caching framework** designed for dLLMs. It leverages token stability across denoising steps, combining **long-interval prompt caching** with **partial response updates** guided by feature similarity. This enables efficient reuse of computations, significantly reducing latency without sacrificing output quality. -->


- **Speedup**: Achieves up to **9.1x** speedup over standard dLLM pipelines, with **no performance loss** on most tasks.
- **Evaluation**: Evaluated on **[LLaDA 8B](https://arxiv.org/abs/2502.09992)** and **[Dream 7B](https://hkunlp.github.io/blog/2025/dream/)**.
- **Latency**: Approaches ARM-level inference speeds in many scenarios.


## :rocket: Pipeline

Here's an overview of the process behind our **dLLM-Cache** method:
![pipeline](./asset/pipeline.png)



## 🛠️ Installation

To get started with dLLM-Cache, follow the installation instructions below.

1. Clone the Repository:
```sh
git clone https://github.com/maomaocun/dLLM-Cache.git
cd dLLM-Cache
```

2. Set Up the Environment:
Create a Python environment with `conda` or `virtualenv` and install dependencies:
```bash
bash install.sh
```


3. Running Experiments:
Run experiments using the provided scripts:

```bash
bash scripts/run_{model_name}_{task_name}_base.sh
```
### :blue_book: Example Usage
1. gsm8k with LLaDA
```bash
bash scripts/run_LLaDA_gsm8k_base.sh
```

2. BBH with Dream
```bash
bash scripts/run_Dream_bbh_base.sh
```


## :postbox: Contact
If you have any questions, please email [yangyicun187@gmail.com](mailto:yangyicun187@gmail.com).


### 🎉 Acknowledgements
This repository was built off of [LLaDA](https://github.com/ML-GSAI/LLaDA), [Dream](https://github.com/HKUNLP/Dream) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## :pushpin: Citation
If you find dLLM-Cache useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{liu2025dllm,
      title={dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching}, 
      author={Zhiyuan Liu and Yicun Yang and Yaojie Zhang and Junjie Chen and Chang Zou and Qingyan Wei and Shaobo Wang and Linfeng Zhang},
      year={2025},
      url={https://github.com/maomaocun/dLLM-cache},
}
```
