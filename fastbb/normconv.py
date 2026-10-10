"""fastbb.normconv: fused convolutions for RAFT-family encoders (csrc/norm_conv.cu; CUTLASS implicit GEMM, fp16 accumulate,
compiled once per source/ABI hash into the user cache, SASS sm_80/86/89/120 + PTX; any input size, no per-size work).

    from fastbb import normconv as nc
    y = nc.conv(x, w, scale=s, shift=t, act=1)                        # act(conv * s + t)          (BN folded into s / t)
    y = nc.conv(x, w, scale=s, shift=t, act=1, res=r, res_mode=1)     # relu(r + relu(conv*s + t)) (ResidualBlock tail)
    y, st = nc.conv_in(x, w)                                          # raw conv + instance-norm stats (batch 1)
    y2, st2 = nc.conv_in(y, w2, pre=st)                               # conv(relu(IN(y))) + stats: center before convolution
    o = nc.in_apply(y2, st2, mode=1, r=x)                             # relu(relu(IN(y2)) + x)

st = (st32 [2, C] fp32 rows rstd / mean, st16 [2, C] fp16 rows scale / shift). x, w, r: fp16 channels-last; C % 8 == 0, K % 8 == 0.
act: 0 none, 1 relu, 2 relu6, 3 tanh, 4 sigmoid, 5 leaky relu (0.01).
"""
import hashlib
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "csrc", "norm_conv.cu")
CUTLASS = os.environ.get("CUTLASS_DIR", "")   # CUTLASS >= 4.x checkout (only needed to build the torch-op .so; see README)
_loaded = [False]


def load():
    if _loaded[0]:
        return True
    abi = f"{torch.__version__}:{torch.version.cuda}:{torch._C._GLIBCXX_USE_CXX11_ABI}"
    tag = hashlib.sha1(open(SRC, "rb").read() + abi.encode()).hexdigest()[:12]
    bdir = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "fastbb", f"normconv_{tag}")
    so = os.path.join(bdir, f"fastbb_normconv_{tag}.so")
    if os.path.exists(so):  # cached: no compiler / CUTLASS needed
        torch.ops.load_library(so)
        _loaded[0] = True
        return True
    if not CUTLASS or not os.path.isdir(os.path.join(CUTLASS, "include")):
        raise RuntimeError("no prebuilt torch-op library for this torch/CUDA build: set CUTLASS_DIR to a CUTLASS >= 4.x checkout and CUDA_HOME to a CUDA 12.8+/13 toolkit (nvcc) to build it, see README")
    os.makedirs(bdir, exist_ok=True)
    from torch.utils.cpp_extension import load as _load
    libdir = os.path.join(bdir, "lib")
    os.makedirs(libdir, exist_ok=True)
    cudart = os.path.join(os.environ.get("CUDA_HOME", ""), "lib", "libcudart.so.13")
    if os.path.exists(cudart) and not os.path.exists(os.path.join(libdir, "libcudart.so")):
        os.symlink(cudart, os.path.join(libdir, "libcudart.so"))
    arch = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in (80, 86, 89, 120)] + ["-gencode=arch=compute_80,code=compute_80"]
    _load(name=f"fastbb_normconv_{tag}", sources=[SRC], build_directory=bdir, verbose=False, is_python_module=False,
          extra_include_paths=[os.path.join(CUTLASS, "include")], extra_ldflags=[f"-L{libdir}"],
          extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr", "-U__CUDA_NO_HALF_OPERATORS__",
                             "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__", *arch])
    _loaded[0] = True
    return True


def ok(x, w):
    """Shapes the kernels take (else: caller's fallback)."""
    return (x.dtype == torch.half and w.dtype == torch.half and x.shape[1] % 8 == 0 and w.shape[0] % 8 == 0
            and w.shape[1] == x.shape[1])


def conv(x, w, scale=None, shift=None, res=None, act=0, res_mode=0, padding=None, stride=1, pre=None, stats=False, cfg=-1):
    load()
    R = w.shape[2]
    p = [R // 2, w.shape[3] // 2] if padding is None else ([padding] * 2 if isinstance(padding, int) else list(padding))
    s = [stride] * 2 if isinstance(stride, int) else list(stride)
    if pre is not None:
        x = in_apply(x, pre, mode=0)
    ps, pb = None, None
    return torch.ops.fastbb_nc.conv(x, w, scale, shift, res, ps, pb, p, s, act, res_mode, stats, cfg)


def conv_in(x, w, padding=None, stride=1, pre=None, eps=1e-5, cfg=-1):
    """raw conv (no bias: it cancels in the instance norm) + its instance-norm stats; pre = stats of x (normalize + ReLU
    applied to x before convolution). Batch 1."""
    if x.shape[0] != 1:
        raise ValueError("conv_in requires batch size 1")
    y, = conv(x, w, padding=padding, stride=stride, pre=pre, stats=False, cfg=cfg)
    K = y.shape[1]
    if K <= 2048:  # stable one-read statistics kernel (fastbb.instnorm), no fp32 copy of y
        from . import kernels as _k
        from .instnorm import _stats_kernel
        st32 = torch.empty((2, K), device=y.device, dtype=torch.float32)
        st16 = torch.empty((2, K), device=y.device, dtype=torch.float16)
        _stats_kernel(_k.Kernels.get(), y, eps, (st32, st16))
        return y, (st32, st16)
    variance, mean = torch.var_mean(y.float(), dim=(0, 2, 3), correction=0)
    rstd = torch.rsqrt(variance + eps)
    return y, (torch.stack((rstd, mean)), torch.stack((rstd, -mean * rstd)).half())


def in_apply(c, st, mode=0, r=None, rst=None, inplace=False):
    load()
    op = torch.ops.fastbb_nc.in_apply_ if inplace else torch.ops.fastbb_nc.in_apply  # in place: declared mutating op
    return op(c, st[0], r, None if rst is None else rst[0], mode)
