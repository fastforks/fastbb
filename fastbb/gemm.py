"""F.linear with chunked fp16 accumulation (prebuilt kernel, any M): fp16-accumulate tensor-core rate, error close to an
fp32-accumulate GEMM (no fp16 sum longer than 32 * chunk products), with fused epilogues: bias, GELU (erf), residual +
LayerScale (out = res + gamma * y). Falls back to torch for unsupported shapes (K % 32, strides % 8, non-fp16, CPU).

    from fastbb.gemm import linear
    y = linear(x, fc1.weight, fc1.bias, act="gelu")                 # x [..., K] fp16
    x = linear(h, fc2.weight, fc2.bias, res=x, gamma=ls2.gamma)     # residual fused (new tensor)
variant: "c2" (chunk 64, default) | "c1" | "c4" | "h" (plain fp16 accumulate) | "f32" (fp32 accumulate MMAs)
"""
import torch
import torch.nn.functional as F

from . import kernels as _k

_ACT = {None: 0, "none": 0, "gelu": 1}


def _ref(x, w, b, act, res, gamma):
    y = F.linear(x, w, b)
    if act == "gelu":
        y = F.gelu(y)
    if res is not None:
        y = torch.addcmul(res, y, gamma) if gamma is not None else res + y
    return y


BN_DEFAULT = 256


def linear(x, w, b=None, act=None, res=None, gamma=None, variant="c2", out=None, bn_pref=None):
    if act not in _ACT:
        raise ValueError(f"unsupported activation: {act}")
    if out is not None and (out.shape != (*x.shape[:-1], w.shape[0]) or out.device != x.device
                            or out.dtype != x.dtype or not out.is_contiguous() or (out.is_cuda and out.data_ptr() % 16)):
        raise ValueError("out must be contiguous with the output shape, input dtype and device")
    def fallback():
        y = _ref(x, w, b, act, res, gamma)
        return out.copy_(y) if out is not None else y
    K = x.shape[-1]
    N = w.shape[0]
    ok = (x.is_cuda and x.dtype == torch.float16 and w.dtype == torch.float16 and K % 32 == 0 and N % 8 == 0
          and w.device == x.device and w.shape[1] == K
          and all(t is None or (t.device == x.device and t.dtype == x.dtype and t.is_contiguous()
                               and t.numel() == N) for t in (b, gamma))
          and (res is None or (res.device == x.device and res.shape == (*x.shape[:-1], N)))
          and x.stride(-1) == 1 and w.stride(-1) == 1 and w.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0 and w.data_ptr() % 16 == 0)
    x2 = x.reshape(-1, K) if ok else None
    if ok and (x2.stride(0) % 8 or (res is not None and (res.dtype != torch.float16 or res.stride(-1) != 1))):
        ok = False
    if not ok:
        return fallback()
    M = x2.shape[0]
    if out is None:
        out = torch.empty((*x.shape[:-1], N), device=x.device, dtype=torch.float16)
    r2 = res.reshape(-1, N) if res is not None else None
    if r2 is not None and r2.stride(0) % 2:
        return fallback()
    kk = _k.Kernels.get()
    bn = 256 if (bn_pref or BN_DEFAULT) == 256 and N >= 256 else 128
    kk.launch(f"gemm_tn_{variant}_{bn}", (((N + bn - 1) // bn, (M + 127) // 128, 1), (256, 1, 1)),
              x2, w, out, b if b is not None else 0, r2 if r2 is not None else 0, gamma if gamma is not None else 0,
              M, N, K, x2.stride(0), w.stride(0), N, r2.stride(0) if r2 is not None else 0, _ACT[act])
    return out
