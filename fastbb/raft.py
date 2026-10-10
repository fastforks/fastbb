"""RAFT-style feature / context encoders (RAFT, RAFT-Stereo, IGEV, Selective-IGEV, DEFOM, Stereo Anywhere, MoCha, ...:
BasicEncoder / MultiBasicEncoder / ResidualBlock with Batch-, Instance-, Group- or no norm), general (any size).

    from fastbb.raft import optimize_encoder
    optimize_encoder(model.fnet, "fp16")        # in place; model.fnet(x) / model.fnet([x1, x2], ...) as before

What it does:
- eval BatchNorm folded into the conv that feeds it (structural: conv1/norm1, conv2/norm2, conv/norm1 attribute pairs and
  Sequential(conv, bn) downsample branches -- the RAFT block layouts)
- InstanceNorm2d (affine False, no running stats) as an NHWC-friendly fp32-statistics version (no layout copies)
- weights in `dtype` (fp16 | bf16), channels-last; the forward casts every tensor input to dtype + channels-last and runs
  with autocast off; outputs stay in dtype (channels-last)
- spec "fp16-hacc": dense convs with B*H*W >= 8*224^2 as fp16-accumulate cuDNN convs (fastbb.cudnn_conv)
Specs: "bf16" (plain PyTorch, exportable), "fp16" (plain PyTorch, exportable), "fp16-hacc" (max: + fastbb InstanceNorm
kernels + fused ResidualBlock relu/residual tails), "bf16-ink" / "fp16-ink" (fastbb InstanceNorm kernels only),
"bf16-ink-fuse" / "fp16-ink-fuse" (+ fused tails).
"""
import torch
import torch.nn as nn

from . import cudnn_conv
from .cnn import _sized_forward

_PAIRS = (("conv1", "norm1"), ("conv2", "norm2"), ("conv3", "norm3"), ("conv", "norm1"), ("conv", "norm"), ("conv", "bn"))


def _fold(conv, bn):
    with torch.no_grad():
        s = (bn.weight.float() if bn.affine else 1) / torch.sqrt(bn.running_var.float() + bn.eps)
        b = (bn.bias.float() if bn.affine else 0) - bn.running_mean.float() * s
        w = conv.weight.float() * s.reshape(-1, 1, 1, 1)
        cb = conv.bias.float() * s + b if conv.bias is not None else b
        conv.weight.data = w.to(conv.weight.dtype)
        conv.bias = nn.Parameter(cb.to(conv.weight.dtype))


def fold_bn(module):
    """Fold eval BatchNorm2d into the preceding conv for RAFT block layouts. Returns the number folded."""
    n = 0
    for m in list(module.modules()):
        for c, b in _PAIRS:
            conv, bn = getattr(m, c, None), getattr(m, b, None)
            if type(conv) is nn.Conv2d and type(bn) is nn.BatchNorm2d and bn.track_running_stats and not bn.training:
                _fold(conv, bn)
                setattr(m, b, nn.Identity())
                n += 1
        if isinstance(m, nn.Sequential):
            ch = list(m.children())
            for i in range(len(ch) - 1):
                if type(ch[i]) is nn.Conv2d and type(ch[i + 1]) is nn.BatchNorm2d and ch[i + 1].track_running_stats and not ch[i + 1].training:
                    _fold(ch[i], ch[i + 1])
                    m[i + 1] = nn.Identity()
                    n += 1
    # norm modules shared by reference elsewhere (e.g. ResidualBlock.norm3 also inside downsample) were replaced in the
    # attribute that owns them; a second reference left in a Sequential is caught by the Sequential pass above
    return n


def _instance_norm_nhwc(self, x):
    """InstanceNorm2d(affine=False/True, no running stats) with fp32 statistics, no layout change, no fp32 copy of x."""
    hw = x.shape[-1] * x.shape[-2]
    m = x.mean((2, 3), keepdim=True, dtype=torch.float32)
    sq = torch.linalg.vector_norm(x, 2, dim=(2, 3), keepdim=True, dtype=torch.float32).square_().div_(hw)
    rstd = torch.rsqrt((sq - m * m).clamp_(min=0) + self.eps)
    if self.affine:
        rstd = rstd * self.weight.float().view(1, -1, 1, 1)
        shift = self.bias.float().view(1, -1, 1, 1) - m * rstd
    else:
        shift = -m * rstd
    return torch.addcmul(shift.to(x.dtype), x, rstd.to(x.dtype))


def _instance_norm_kernel(self, x, mode=0, res=None):
    """fastbb kernels: fp32 stats (one read, atomics) + one fused normalize pass, NHWC, no copies. Falls back for other layouts.
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
    st = torch.zeros((N, C, 2), device=x.device, dtype=torch.float32)
    R = max(1, 256 // (C // 8))
    nbx = max(1, min((HW + R - 1) // R, max(1, 288 // N)))
    rpb = (HW + nbx - 1) // nbx
    kk.launch(f"in_stats_{sfx}", ((nbx, N, 1), (256, 1, 1)), x, st, HW, C, rpb)
    y = torch.empty_like(x)
    w = self.weight.float().contiguous() if self.affine else None
    b = self.bias.float().contiguous() if self.affine else None
    kk.launch(f"in_apply_{sfx}", kk.linear(N * HW * C // 8), x, y, st, w if w is not None else 0, b if b is not None else 0,
              res if res is not None else 0, N, HW, C, float(self.eps), mode)
    return y


def _resblock_in_forward(self, x):
    """RAFT ResidualBlock (instance norm) with the ReLUs and the residual add fused into the norm passes."""
    y = _instance_norm_kernel(self.norm1, self.conv1(x), 1)
    r = x if self.downsample is None else _instance_norm_kernel(self.norm3, self.downsample[0](x), 0)
    return _instance_norm_kernel(self.norm2, self.conv2(y), 2, r.contiguous(memory_format=torch.channels_last))


def _is_raft_resblock_in(m):
    if not all(type(getattr(m, n, None)) is nn.InstanceNorm2d and not getattr(m, n).track_running_stats for n in ("norm1", "norm2")):
        return False
    if not all(type(getattr(m, n, None)) is nn.Conv2d for n in ("conv1", "conv2")) or not isinstance(getattr(m, "relu", None), nn.ReLU):
        return False
    d = getattr(m, "downsample", "missing")
    if d is None:
        return True
    return isinstance(d, nn.Sequential) and len(d) == 2 and type(d[0]) is nn.Conv2d and d[1] is getattr(m, "norm3", None) \
        and type(d[1]) is nn.InstanceNorm2d and not d[1].track_running_stats


def _cast(v, dt):
    if torch.is_tensor(v) and v.is_floating_point():
        return v.to(dt).contiguous(memory_format=torch.channels_last) if v.dim() == 4 else v.to(dt)
    if isinstance(v, (list, tuple)):
        return type(v)(_cast(t, dt) for t in v)
    return v


def _keep_forward(orig, dt_out):
    def f(x, *a, **kw):
        return orig(_cast(x, torch.bfloat16), *a, **kw).to(dt_out)
    return f


def optimize_encoder(enc, spec="fp16", keep_bf16=(), cast_args=None):
    """In place. Returns info dict.
    keep_bf16: names of direct submodules that stay bf16 (their input is cast to bf16, output to the encoder dtype) -- for
      inputs / activations beyond the fp16 range (e.g. DEFOM's convs on DPT features: |x| up to 2.5e5 / 1.8e6).
    cast_args: indices of positional args cast to the encoder dtype (default: all); others pass through unchanged."""
    flags = set(spec.split("-"))
    dt = torch.bfloat16 if "bf16" in flags else torch.float16
    enc.eval()
    kept = [getattr(enc, n) for n in keep_bf16] if dt == torch.float16 else []
    kept_ids = {id(m) for k in kept for m in k.modules()}
    info = {"folded_bn": fold_bn(enc), "instance_norm": 0, "hacc_convs": 0}
    for m in enc.modules():
        if type(m) is nn.InstanceNorm2d and not m.track_running_stats:
            m.forward = (_instance_norm_kernel if "ink" in flags or "hacc" in flags else _instance_norm_nhwc).__get__(m)
            info["instance_norm"] += 1
    if "hacc" in flags or "fuse" in flags:  # RAFT residual blocks with instance norm: fused relu / residual tails
        for m in enc.modules():
            if type(m).__name__ in ("ResidualBlock",) and _is_raft_resblock_in(m) and m.forward.__func__ is type(m).forward:
                m.forward = _resblock_in_forward.__get__(m)
                info["fused_blocks"] = info.get("fused_blocks", 0) + 1
    enc.to(dt)
    for k in kept:
        k.to(torch.bfloat16)
        k.forward = _keep_forward(k.forward, dt)
    for m in enc.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
    if "hacc" in flags and dt == torch.float16:
        info["hacc_convs"] = cudnn_conv.patch_convs(enc)  # fp16 weights only: the kept bf16 modules are skipped
        for m in enc.modules():
            if isinstance(m, nn.Conv2d) and "forward" in m.__dict__ and id(m) not in kept_ids:
                kh, kw_ = m.kernel_size
                # narrow convs (< 64 in/out channels: EfficientNet / MobileNet stages) are memory-bound: no gain, slower plans
                m._macs_per_px = 0 if min(m.in_channels, m.out_channels) < 64 else \
                    m.in_channels * m.out_channels * kh * kw_ / (m.stride[0] * m.stride[1])
                m.forward = _sized_forward.__get__(m)
    orig = enc.forward

    from .cnn import _STATE

    def forward(*a, **kw):
        _STATE["hacc"] = True  # encoder inputs are full images: per-conv MAC rule only
        with torch.autocast("cuda", enabled=False):
            a = tuple(_cast(v, dt) if cast_args is None or i in cast_args else v for i, v in enumerate(a))
            return orig(*a, **({k: _cast(v, dt) for k, v in kw.items()} if cast_args is None else kw))
    enc.forward = forward
    enc._fastbb_info = info
    return info
