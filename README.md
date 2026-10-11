# fastbb

Fast, general (any input size, nothing compiled or tuned per size) drop-in backbones for PyTorch inference: DINOv2-style ViTs,
CNNs (timm / torchvision) and RAFT-style feature encoders. Used by the `cuda-kernels` branches of the
[fastforks](https://github.com/fastforks) stereo / depth forks. Two tiers everywhere: **friendly** (plain PyTorch ops, exports to
ONNX / TensorRT) and **max** (custom CUDA kernels, cuDNN fp16-accumulate convolutions, CUTLASS fused convs; eager only).

## Performance (RTX 5060 Ti 16 GB, sm_120, torch 2.14.1+cu130, timm 1.0.30, 2026-10-10; batch 1, median of 10-20 warm calls)

| model / input | upstream | friendly | max | notes |
|---|---|---|---|---|
| DINOv2 ViT-L, 896x1344 | 175.6 ms | 143.8 ms | 114.0 ms (`hh-g1`) | parity (rel. L2 of patch tokens vs fp32) < 0.01 |
| ResNet-50, 224x224 | 3.46 ms | - | 0.62 ms (trace + CUDA graph) | b32: 16.3 -> 6.6 ms |
| ResNet-50, 1024x1024 | 9.27 ms | - | 4.29 ms | |
| ViT-B/16 (timm), 1024 px | 32.3 ms | - | 21.6 ms (`install_vit_blocks`) | |

Numbers: one GPU, median of warm calls; peak memory, first call and new-size cost were measured per row.
One GPU only; fp16-accumulating convs/GEMMs are a GeForce
(consumer Blackwell/Ada) win and shrink on A100/H100 (use the `g0` / non-`hacc` specs there).

## Install

```bash
pip install -e .            # pure Python + CUDA sources; needs torch >= 2.4 with CUDA
pip install nvidia-cudnn-frontend   # optional: fp16-accumulate cuDNN convs (the "hacc" / "max" CNN and encoder paths)
```

## Prebuilt kernels and building

- `fastbb/prebuilt/fastbb_kernels_*.fatbin` (11 MB, in the repo): attention / GEMM / norm kernels as a driver-API fatbin with SASS for
  sm_80/86/87/89/90/110/120 + compute_80 PTX; loaded through ctypes, independent of the torch version. No compiler needed.
  Rebuilt automatically (needs `nvcc` from CUDA >= 12.8, `CUDA_HOME`) when the `.cu` source changes: `python -m fastbb.kernels`.
- The CUTLASS torch-op libraries (`csrc/fused_conv.cu`, `csrc/norm_conv.cu`; used by the CNN `fused` spec and the RAFT-style
  `normconv` encoders) are tied to the torch ABI and are therefore **not** shipped: they are built on first use into
  the user cache (`~/.cache/fastbb/` by default). Needs a CUTLASS >= 4.x checkout (`CUTLASS_DIR=/path/to/cutlass`, header-only) and
  nvcc (`CUDA_HOME`; the pip packages `nvidia-cuda-nvcc` + `nvidia-cuda-runtime` of a CUDA 13 torch build are enough).
  Without them the code falls back to the other tiers.
- Only tested on sm_120 (RTX 5060 Ti) so far; the other SASS targets are included but have not been executed on their target GPUs.

## API (fastbb)
```python
import fastbb
# ViT, DINOv2-style module inside a model (official dinov2, DAv2 copies, MoGe, ...): get_intermediate_layers -> fast path
fastbb.install_vit(model.pretrained, "hh-g1")          # fp16 weights in place
# any ViT block stack (timm VisionTransformer / DeiT / DINOv2 timm / official): block-level drop-in, tokenizer stays the model's
fastbb.install_vit_blocks(timm_vit, "fh-g1")           # converts to fp16 (convert=False keeps the dtype)
# standalone DINOv2 from the official checkpoints (fp16 on the CPU before the move; "-graph" suffix = CUDA graph per shape)
from fastbb.models.dinov2 import dinov2
m = dinov2("vitl14", "path/to/dinov2_vitl14_pretrain.pth", spec="hh-g1")
# CNN backbones (timm / torchvision): friendly "fold" | max "fold-gacc-fused-trace" (default) | "+graph"
from fastbb.cnn import optimize; net = optimize(timm.create_model("resnet50", pretrained=True), "fold-gacc-fused-trace")
# RAFT-style encoders / DPT heads / any full-res conv stack: bf16 | fp16 (friendly) | fp16-hacc (max: IN kernels, fused tails)
fastbb.optimize_convnet(model.fnet, "fp16-hacc", keep_bf16=("convd",), cast_args=(0,))
# CUDA graph for any callable (opt-in, per input shape)
from fastbb.graphs import graphed
```
ViT spec `<attn>-g<n>`: attn `sdpa` (friendly) | `fh` (flash64, fp32 QK^T, fp16-acc PV) | `hh` (fp16-acc both) | `ff`; `g0` fp32-acc
GEMMs, `g1` fp16-acc GEMMs, `g2` all but fc2, `g3` chunked fp16-acc GEMM kernel (~fp32 error, between g0 and g1 in speed).
Rules of thumb (measured): stereo/mono ViT-L at >= 518 px: `hh-g1` (validate on the model; SA prefers `fh-g1`); B*N < 400 tokens
falls back to SDPA automatically; small inputs at batch 1 are CPU-bound -> `-graph` (ViT) / `trace` or `graph` (CNN).
CNN: hacc only when B*H*W of the network input >= 8*224^2 and >= 64 channels; fused CUTLASS convs
use `torch.ops.fastbb.*`. Weights go to fp16 on the CPU before the move. Rebuild the shipped fatbin with
`python -m fastbb.kernels`; CUTLASS libraries build on first use when `CUTLASS_DIR` and `CUDA_HOME` are set.

## Licence

Apache-2.0 (see `LICENSE`, `NOTICE`). Contains CUTLASS-based kernels (BSD-3-Clause, `third_party/CUTLASS_LICENSE.txt`) and code
following DINOv2 (Apache-2.0, `third_party/DINOv2_LICENSE`). No model weights are included.
