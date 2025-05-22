# dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching

Official PyTorch implementation of the paper **["dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching"](./asset/dLLM-Cache_paper.pdf)** (dLLM-Cache).

## :fire: News

- [2025/05/22] Our paper has been released.  



## ✨️ Key Highlights
<!-- Our approach excels across diverse tasks, as shown in the radar chart below: -->
![radar_speed](./asset/radar.png)

<!-- Diffusion-based Large Language Models (dLLMs) offer a robust alternative to Autoregressive Models (ARMs) by iteratively denoising masked text segments. However, their bidirectional attention mechanism results in high inference latency, making traditional ARM acceleration methods like Key-Value caching incompatible.

**dLLM-Cache** is a **training-free adaptive caching framework** designed for dLLMs. It leverages token stability across denoising steps, combining **long-interval prompt caching** with **partial response updates** guided by feature similarity. This enables efficient reuse of computations, significantly reducing latency without sacrificing output quality. -->


- **Speedup**: Up to **9.1x faster** inference compared to standard dLLM pipelines.
- **Performance**: Validated on **LLaDA 8B** and **Dream 7B**.
- **Latency**: Approaches ARM-level inference speeds in many scenarios.


## :rocket: Pipeline

Here's an overview of the process behind our **dLLM-Cache** method:
![pipeline](./asset/pipeline.png)



## :postbox: Contact
If you have any questions, please email [yangyicun187@gmail.com](mailto:yangyicun187@gmail.com).



## :pushpin: Citation
If you find dLLM-Cache useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{liu2025dllm,
      title={dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching}, 
      author={Zhiyuan Liu and Yicun Yang and Yaojie Zhang and Junjie Chen and Chang Zou and Qingyan Wei and Shaobo Wang and Linfeng Zhang },
      year={2025},
      url={https://github.com/maomaocun/dLLM-Cache},
}
```