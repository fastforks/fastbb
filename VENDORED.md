# Vendored code

This file lists every fork that carries a copy of fastbb code: update the row when a copy is added, moved or trimmed differently.

The forks carry the operators they use on `cuda-kernels`, imported as `fastbb_vendored`. Their `portable` branches do not use fastbb.

Edit the sources here, then run `tools/sync_vendored.sh [raft|crestereo|sam3|all]`. `RAFT_DIR`, `CRESTEREO_DIR` and `SAM3_DIR` override the sibling checkout locations. The destination must be on `cuda-kernels` with a clean vendor directory. The script shows the resulting diff and does not commit it.

| Fork (repo) | Copy in the fork | Sources under `fastbb/` | Use |
|---|---|---|---|
| RAFT-Stereo (fastforks/raft) | `fastbb_vendored/` | `normconv.py`, `csrc/norm_conv.cu` | Context-encoder convolution and folded BatchNorm |
| CREStereo-Pytorch (fastforks/crestereo) | `fastbb_vendored/` | `instnorm.py`, `kernels.py`, `fastbb_kernels.cu` | Default instance normalization |
| SAM 3 (fastforks/sam3) | `sam3/fastbb_vendored/` | `attention.py`, `kernels.py`, `fastbb_kernels.cu`, `cudnn_conv.py` | ViT flash attention (head_dim 64, fp16 accumulation), fp16-accumulate cuDNN convolutions |

Python headers and the generated `__init__.py` record each source file's last-changing commit. `trim_encoder_vendor.py` keeps only convolution/BatchNorm operators in the RAFT copy, selects only instance-normalization kernels and their loader entries for CREStereo, and only the head_dim-64 flash-attention kernels for SAM 3 (fatbin about 1.1 MiB). Syncing CREStereo or SAM 3 requires nvcc to rebuild that subset: its fatbin is about 470 KiB, rather than the full library's 11 MiB. Source hashes determine binary filenames. Users can load the shipped fatbin without compiling it; the optional convolution extension still builds on first use.
