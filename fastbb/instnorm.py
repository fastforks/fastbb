"""InstanceNorm for NHWC fp16 / bf16 activations (used by fastbb.raft and vendored into the RAFT-family forks).

_instance_norm_nhwc: plain PyTorch (exportable), stable statistics, no fp32 copy of the activation in eager inference.
_instance_norm_kernel: fastbb CUDA kernels (kernels.py): fp32 statistics in one read (per-thread shifted sums merged with Chan's
formula in a fixed order: stable and deterministic) + one fused normalize pass; falls back to _instance_norm_nhwc for other
layouts / dtypes. Both are bound to an nn.InstanceNorm2d as `forward` (first argument = the module).
"""
import torch


def _instance_norm_ref(self, x):
    """(x - mean) * rstd in fp32 (export / autograd path; materializes fp32 temporaries)."""
    xf = x.float()
    variance, mean = torch.var_mean(xf, dim=(2, 3), correction=0, keepdim=True)
    y = (xf - mean) * torch.rsqrt(variance + self.eps)
    if self.affine:
        y = y * self.weight.float().view(1, -1, 1, 1) + self.bias.float().view(1, -1, 1, 1)
    return y.to(x.dtype)


def _affine32(self, x):
    """fp32 weight / bias (no copy for fp32 parameters); ones / zeros without affine (cached per module and device)."""
    if self.affine:
        return self.weight.detach().float(), self.bias.detach().float()
    c = self.__dict__.get("_fbb_wb")
    if c is None or c[0].device != x.device:
        C = x.shape[1]
        c = self.__dict__["_fbb_wb"] = (torch.ones(C, device=x.device), torch.zeros(C, device=x.device))
    return c


def _instance_norm_nhwc(self, x):
    """InstanceNorm2d(affine=False/True, no running stats). Eager inference: fp32 mean + Welford std (fp32 accumulation) read
    from the fp16 activation, then (x - mean) * rstd * w + b in fp32 arithmetic by PyTorch's native eval batch-norm kernel (per-instance
    statistics as its running statistics); output in the input dtype and layout, no fp32 copy. Export / tracing / autograd
    use the plain fp32 formula (_instance_norm_ref)."""
    if (not x.is_cuda or torch.jit.is_tracing() or torch.onnx.is_in_onnx_export() or torch.compiler.is_compiling() or x.dtype == torch.float32
            or (torch.is_grad_enabled() and (x.requires_grad or (self.affine and self.weight.requires_grad)))):
        return _instance_norm_ref(self, x)
    sd, m16 = torch.std_mean(x, (2, 3), correction=0)                 # Welford, fp32 accumulation (exact for constants)
    mean = x.sum((2, 3), dtype=torch.float32) / (x.shape[2] * x.shape[3])
    mean = torch.where(sd == 0, m16.float(), mean)                      # constant channel: the sum may round, Welford not
    var = sd.float().square_()
    w, b = _affine32(self, x)
    y = torch.empty_like(x)  # native kernel (cuDNN's folds scale / shift: cancellation for large means)
    e1, e2 = torch.empty(0, device=x.device), torch.empty(0, device=x.device)
    for i in range(x.shape[0]):
        torch.ops.aten.native_batch_norm.out(x[i:i + 1], w, b, mean[i], var[i], False, 0.0, self.eps,
                                             out=y[i:i + 1], save_mean=e1, save_invstd=e2)
    return y


def _stats_kernel(kk, x, eps, nc=None):
    """st [N, C, 2] fp32 = (mean, rstd) per (n, c) of an NHWC fp16 / bf16 activation (C % 8 == 0, C <= 2048).
    nc: optional (fp32 [2, C], fp16 [2, C]) also filled in fastbb.normconv's layout (batch 1)."""
    N, C, H, W = x.shape
    HW = H * W
    R = max(1, 256 // (C // 8))
    nbx = max(1, min((HW + R - 1) // R, max(1, 288 // N)))
    rpb = (HW + nbx - 1) // nbx
    buf = torch.empty(2 * N * C * (nbx + 1), device=x.device, dtype=torch.float32)
    partial, st = buf[:2 * N * C * nbx], buf[2 * N * C * nbx:].view(N, C, 2)  # per block (mean, M2); result
    kk.launch_packed(f"in_stats_{'h' if x.dtype == torch.float16 else 'b'}", (nbx, N, 1), (256, 1, 1), x, partial, HW, C, rpb)
    kk.launch_packed("in_stats_final", ((N * C + 7) // 8, 1, 1), (256, 1, 1), partial, st, N, nbx, HW, C, rpb, float(eps),
                     *(nc if nc is not None else (0, 0)))
    return st


def _instance_norm_kernel(self, x, mode=0, res=None):
    """fastbb kernels: fp32 stats in one read (stable, deterministic) + one fused normalize pass, NHWC, no copies. Falls back for other layouts.
    mode 0: norm, 1: relu(norm), 2: relu(res + relu(norm)) (RAFT ResidualBlock tail)."""
    N, C, H, W = x.shape
    if not (x.is_cuda and x.dtype in (torch.float16, torch.bfloat16) and C % 8 == 0 and C <= 2048
            and x.is_contiguous(memory_format=torch.channels_last) and x.data_ptr() % 16 == 0
            and (res is None or (res.dtype == x.dtype and res.shape == x.shape and res.data_ptr() % 16 == 0
                                 and res.is_contiguous(memory_format=torch.channels_last)))):
        y = _instance_norm_nhwc(self, x)
        if mode >= 1:
            y = y.relu_()
        return (y + res).relu_() if mode == 2 else y
    from . import kernels as _k
    kk = _k.Kernels.get()
    sfx = "h" if x.dtype == torch.float16 else "b"
    HW = H * W
    st = _stats_kernel(kk, x, self.eps)
    y = torch.empty_like(x)
    w = self.weight.float().contiguous() if self.affine else None
    b = self.bias.float().contiguous() if self.affine else None
    kk.launch_packed(f"in_apply_{sfx}", *kk.linear(N * HW * C // 8), x, y, st, w if w is not None else 0,
                     b if b is not None else 0, res if res is not None else 0, N, HW, C, float(self.eps), mode)
    return y
