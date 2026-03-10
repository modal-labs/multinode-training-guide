> [!IMPORTANT]
> Our multi-node cluster training product is in early preview and not generally accessible. Please [**contact us**](https://modal.com/slack) for access.

---


<p align="center">
  <a href="https://modal.com">
    <img src="https://modal-public-assets.s3.amazonaws.com/bigicon.png" height="96">
    <h2 align="center">Modal Multinode Training Guide</h2>
  </a>
</p>

Well documented examples of running distributed training jobs on [Modal](https://modal.com).
Use this repository to learn how to build distributed training jobs on Modal.

# Examples

- [**`benchmark/`**](/benchmark/) contains performance and reliability testing, using AWS EFA by default.
- [**`lightning/`**](/lightning/) a simple lightning.ai Fabric example.
- [**`nanoGPT/`**](/nanoGPT/) training Karpathy's nanoGPT reproduction of OpenAI's GPT-2.
- [**`resnet50/`**](/resnet50/) training a ResNet50 model on the ImageNet dataset.
- [**`starcoder/`**](/starcoder) accelerated finetuning of Llama-2-7B on Rust and Go code, supporting either `torchrun` or `accelerate`.
- [**`trl/`**](/trl) reinforcement learning training with GRPO to train a mathematical reasoning model using the `trl` library with sandboxed code execution.

# Documentation

The multi-node training guide is currently available on Notion: [modal-com.notion.site/Multi-node-docs](https://modal-com.notion.site/Multi-node-docs-1281e7f16949806f966adedfe8b2cb74?pvs=4).

Other relevant documentation in our guide:

- [Multi-GPU Training](https://modal.com/docs/guide/gpu#multi-gpu-training)
- [Using CUDA on Modal](https://modal.com/docs/guide/cuda)
- [GPU Metrics](https://modal.com/docs/guide/gpu-metrics)

# Demo

[demo video of launching resnet50 on 32 H100s](https://github.com/user-attachments/assets/ed3dc6fe-61f2-4abc-ab48-5b5d01f65c31)

## License

The [MIT license](LICENSE).
