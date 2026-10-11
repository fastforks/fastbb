# Vendored encoder code

The stereo forks carry the encoder operators they use on `cuda-kernels`, imported as `fastbb_vendored`. Their `portable` branches do not use fastbb.

Edit the sources here, then run `tools/sync_vendored.sh [raft|crestereo|all]`. `RAFT_DIR` and `CRESTEREO_DIR` override the sibling checkout locations. The destination must be on `cuda-kernels` with a clean vendor directory. The script shows the resulting diff and does not commit it.

| Fork | Sources under `fastbb/` | Use |
|---|---|---|
| RAFT-Stereo | `normconv.py`, `csrc/norm_conv.cu` | Context-encoder convolution and folded BatchNorm |
| CREStereo-Pytorch | `instnorm.py`, `kernels.py`, `fastbb_kernels.cu` | Default instance normalization |

Python headers and the generated `__init__.py` record each source file's last-changing commit. `trim_encoder_vendor.py` keeps only convolution/BatchNorm operators in the RAFT copy, and selects only instance-normalization kernels and their loader entries for CREStereo. Syncing CREStereo requires nvcc to rebuild that subset: its fatbin is about 470 KiB, rather than the full library's 11 MiB. Source hashes determine binary filenames. Users can load the shipped fatbin without compiling it; the optional convolution extension still builds on first use.
