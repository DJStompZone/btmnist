# Balanced Ternary MNIST (btmnist)

This project is a proof-of-concept implementation of **balanced ternary neural networks** for image classification.  
Weights and activations live in {-1, 0, +1} with learned per-channel scaling factors, and we accelerate linear layers with a custom **Triton popcount kernel** that packs ternary values into two bitplanes.

The goal: explore whether balanced ternary can be competitive with conventional float32 training and inference, while cutting multiplications down to bitwise ops and a few per-channel scalars.

## Features

- **Balanced ternary quantization**:
  - Straight-Through Estimator (STE) for training.
  - TTQ-style learned positive/negative scales (αₚ, αₙ).
- **Custom Triton kernel**:
  - Bit-pack ternary tensors into two bitplanes.
  - Multiply-free GEMM using AND+popcount.
  - Autograd wrapper with STE backward.
- **PyTorch modules**:
  - `TernaryConv2d`, `TernaryLinear`, `TernaryLinearTriton`.
- **MNIST training**:
  - LeNet-style CNN with ternary everywhere.
  - Switch between PyTorch and Triton FC layers via CLI flag.
- **Export + inference**:
  - Save ternary state (int8 weights + float scales).
  - Reload and run evaluation with Triton-only linear layers.

## Install

```bash
git clone https://github.com/djstompzone/btmnist.git
cd btmnist
poetry install
````

Dependencies:

- Python 3.11–3.13
- [PyTorch](https://pytorch.org/) with CUDA
- [Triton](https://github.com/triton-lang/triton)

## Usage

### Train

By default training uses pure PyTorch ternary layers:

```bash
poetry run python -m btmnist.train --epochs 8 --batch 128 --lr 2e-3 --device cuda
```

To use the **Triton autograd linear layers** (forward = popcount kernel, backward = STE surrogate):

```bash
poetry run python -m btmnist.train --epochs 8 --batch 128 --lr 2e-3 \
    --device cuda --use-triton-linear
```

Exports a ternary state dict after training (default `bt_mnist_export.pt`).

### Inference

Run evaluation with Triton linear layers only:

```bash
poetry run python -m btmnist.inference --export bt_mnist_export.pt --device cuda
```

## Project Layout

```plaintext
btmnist/
│
├── pyproject.toml        # Poetry metadata
└── btmnist/
    ├── __init__.py
    ├── quant.py          # Quantization + ternary modules
    ├── kernel.py         # Triton kernel + autograd wrapper
    ├── train.py          # MNIST training script
    └── inference.py      # Export + Triton inference
```

## TODO

- More hyperparameter tuning (αₚ/αₙ init, learning rate, etc)
- Try per-channel thresholds Δ
- Add ternary gradient compression (TernGrad-style)
- Adapt popcount kernels from XNOR-Net for even lower overhead
- Test on larger datasets (CIFAR-10, ImageNet)
- Test on CPU (AVX2/AVX512 bitwise (tritwise?) ops) [¹](https://news.ycombinator.com/item?id=39550178)

## Notes

- **BatchNorm**: can be used during training, then folded into αₚ/αₙ and bias for inference.
- **Convs**: currently PyTorch only, but they can be mapped to Triton via im2col + GEMM or a direct conv kernel.

## References

- *"Ternary Weight Networks"*; Li, F., Zhang, B., & Liu, B. (2016) [Link](https://arxiv.org/abs/1605.04711)
- *"Trained Ternary Quantization"*; Chenzhuo Zhu, Song Han, Huizi Mao, William J. Dally (2017) [Link](https://arxiv.org/abs/1612.01064)
- *"FATNN: Fast and Accurate Ternary Neural Networks"*; Peng Chen, Bohan Zhuang, Chunhua Shen (2021) [Link](https://arxiv.org/abs/2008.05101)
- *"Ternary is the New Binary"*; USN Ternary Research Group (web article, 2022) [Link](https://www.ternaryresearch.com/)
- *"MNIST Ternary PyTorch implementation"*; Vinay Sisodia (2018) [Link](https://github.com/vinsis/ternary-quantization)
- *"High Efficiency Multiply-Accumulator Using Ternary Logic and Ternary Approximate Algorithm"*; W. Wen et al. (2025); (DOI: 10.1109/TCSI.2024.3492797) [Link](https://ieeexplore.ieee.org/document/10755970)
- *"TRQ: Ternary Neural Networks With Residual Quantization"*; Li, Y., Ding, W., Liu, C., Zhang, B., & Guo, G. (2021); (DOI: 10.1609/aaai.v35i10.17036) [Link](https://ojs.aaai.org/index.php/AAAI/article/view/17036)
- *"Ternary Neural Networks for Efficient Biometric Data Analysis"*; Giacomo Agnetti (2024) [Link](http://webthesis.biblio.polito.it/id/eprint/33913)

## License

MIT License. See the [LICENSE](LICENSE) file for details.
