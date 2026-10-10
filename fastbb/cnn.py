"""General fast inference for CNN-style backbones (timm / torchvision ResNet, ConvNeXt V1/V2, EfficientNet, MobileNet,
RegNet, Swin's linears, ...): one call on an eval model, no per-size compilation.

    from fastbb.cnn import optimize
    model = optimize(timm.create_model("resnet50", pretrained=True), spec="hacc")
    y = model(x)            # x fp32/fp16 NCHW (any size); runs fp16 channels-last inside, returns what the model returns

What it does (each step optional via spec flags, "-" separated: "fp16" < "fold" < "hacc" < "gacc"; "graph" = opt-in CUDA
graph per input shape, captured on first use):
- fold : eval BatchNorm folded into the preceding conv (only when an FX graph proves a single-use Conv -> BN edge;
         timm BatchNormAct2d keeps its activation), then fp16 weights + channels-last
- hacc : dense convs (groups 1) as fp16-accumulate cuDNN convs (precompiled engines only; fastbb/cudnn_conv.py)
- gacc : nn.Linear (ConvNeXt MLPs, Swin, heads) as fp16-accumulate cuBLAS GEMMs
- fuse : timm ConvNeXt blocks: MLP through fastbb.gemm (chunked fp16 acc) with fused GELU and LayerScale+residual
- fused: exp 008's CUTLASS implicit-GEMM convs with fused bias + residual + activation (fp16 acc, fixed tile rule), depthwise /
         pool / stem (RGB padded to 8) kernels, fx-flattened graph (fastbb/fused.py, cached extension); "trace": TorchScript
         trace of the result (C++ executor, no fusers) -> lower CPU cost at batch 1
Default spec "fold-gacc-fused-trace" (max); friendly = "fold" (plain PyTorch, exportable). Activations stay as the model computes them (fp16).
"""
import contextlib
import warnings
import weakref

import torch
import torch.nn as nn

from . import cudnn_conv


def _fold_bn(model, sample):
    """Fold eval BatchNorm2d into the conv whose output it directly consumes. Returns the number folded."""
    # A runtime hook cannot see another consumer of the convolution output.
    # Fold only graph-proven, single-use Conv -> BN edges.
    from collections import Counter

    import torch.fx as fx

    class Tracer(fx.Tracer):
        def is_leaf_module(self, m, name):
            return isinstance(m, nn.BatchNorm2d) or super().is_leaf_module(m, name)

    try:
        graph = Tracer().trace(model)
    except Exception:
        return 0
    modules = dict(model.named_modules())
    calls = Counter(id(modules[n.target]) for n in graph.nodes if n.op == "call_module")
    pairs = []
    for node in graph.nodes:
        if node.op != "call_module" or not node.args:
            continue
        bn = modules[node.target]
        parent = node.args[0]
        if not isinstance(parent, fx.Node) or parent.op != "call_module":
            continue
        conv = modules[parent.target]
        if (type(conv) is nn.Conv2d and isinstance(bn, nn.BatchNorm2d)
                and not bn.training and bn.track_running_stats
                and len(parent.users) == 1 and calls[id(conv)] == calls[id(bn)] == 1):
            pairs.append((conv, bn))
    seen = set()
    n = 0
    for c, bn in pairs:
        if id(bn) in seen or id(c) in seen:
            continue
        seen.update((id(bn), id(c)))
        with torch.no_grad():
            s = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps) if bn.affine else \
                1 / torch.sqrt(bn.running_var.float() + bn.eps)
            b = (bn.bias.float() if bn.affine else 0) - bn.running_mean.float() * s
            w = c.weight.float() * s.view(-1, 1, 1, 1)
            cb = c.bias.float() * s + b if c.bias is not None else b
            c.weight.data = w.to(c.weight.dtype)
            c.bias = nn.Parameter(cb.to(c.weight.dtype))
        act = getattr(bn, "act", None)
        drop = getattr(bn, "drop", None)
        bn._fastbb_folded = True
        if act is not None or drop is not None:  # timm BatchNormAct2d: keep drop + act
            bn.forward = (lambda self, x: self.act(self.drop(x))).__get__(bn)
        else:
            bn.forward = (lambda self, x: x).__get__(bn)
        n += 1
    return n


HACC_MIN_PIXELS = 8 * 224 * 224  # B*H*W of the NETWORK input (rule from exp 008 ResNet-50): below it the whole net is
# launch-bound and the cuDNN-frontend dispatch costs more than the 2x MMA rate saves -> plain torch convs (fp32 accumulate).
# (v1 applied it per conv input, which disabled hacc for every conv after the stem: 1024 px b1 7.1 -> 9.1 ms.)
HACC_MIN_MACS = 2e8  # per conv: tiny convs stay on torch even in large inputs
_STATE = {"hacc": True}


def _sized_forward(self, x):
    n, _, h, w = x.shape
    if _STATE["hacc"] and self.__dict__.get("_macs_per_px", 0) * n * h * w >= HACC_MIN_MACS:
        return cudnn_conv.conv_half_acc(x, self.weight, self.bias, list(self.padding), list(self.stride))
    return torch.nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding)


@contextlib.contextmanager
def _gacc(on):
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    torch.backends.cuda.matmul.allow_fp16_accumulation = on
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old


class FastCNN(nn.Module):
    """Inference wrapper for a fixed optimized architecture. The opt-in trace / CUDA graph bake in the weights: they are rebuilt
    after load_state_dict (any submodule), .to() / .half() / device moves (O(1) check per call); after changing weights in place
    otherwise (p.data.copy_, optimizer steps) call invalidate()."""
    def __init__(self, model, flags):
        super().__init__()
        self.model, self.flags = model, flags
        self.__dict__["_gen"] = 0
        self.__dict__["_p0"] = next(model.parameters(), None)
        ref = weakref.ref(self)

        def loaded(module, incompatible_keys):
            s = ref()
            if s is not None:
                s.invalidate()
        for m in model.modules():
            if m._parameters or m._buffers:
                m.register_load_state_dict_post_hook(loaded)

    def invalidate(self):
        """Drop the trace / CUDA graphs (rebuilt at the next call)."""
        self.__dict__["_gen"] += 1

    def _apply(self, fn, *a, **kw):
        self.invalidate()
        return super()._apply(fn, *a, **kw)

    def forward(self, x, *a, **kw):
        if getattr(self.model, "pad8", False):
            x = torch.ops.fastbb.input_nhwc8(x)  # one kernel: NCHW fp32/fp16 -> NHWC fp16, RGB zero-padded to 8 channels
        else:
            x = x.to(torch.float16).contiguous(memory_format=torch.channels_last)
        _STATE["hacc"] = x.shape[0] * x.shape[2] * x.shape[3] >= HACC_MIN_PIXELS
        with torch.autocast("cuda", enabled=False), _gacc("gacc" in self.flags):
            if a or kw or not ("trace" in self.flags or "graph" in self.flags):
                return self.model(x, *a, **kw)
            d, p0 = self.__dict__, self.__dict__["_p0"]
            key = (d["_gen"], None if p0 is None else (p0.data_ptr(), p0.dtype))
            if d.get("_key") != key:
                d["_key"], d["_tr"], d["_graph"] = key, None, None
            runner = self._traced() if "trace" in self.flags else self.model
            if "graph" in self.flags and x.is_cuda:  # opt-in: CUDA graph per input shape (small inputs: launch-bound)
                g = d["_graph"]
                if g is None:
                    from .graphs import Graphed
                    g = d["_graph"] = Graphed(runner)
                return g(x)
            return runner(x)

    def _traced(self):
        """Shared guarded tracer (fastbb.trace.Traced): one validated TorchScript trace per input shape."""
        tr = self.__dict__.get("_tr")
        if tr is None:
            from .trace import Traced
            tr = self.__dict__["_tr"] = Traced(self.model, name="CNN", shapes=None, key=lambda shapes: tuple(tuple(s) for s in shapes))
        return tr

    def __getattr__(self, name):  # pass-through (forward_features, feature_info, ...)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def _convnext_block_fwd(self, x):
    """timm ConvNeXtBlock (V1 / V2, NHWC MLP path): fc1 + GELU and fc2 + LayerScale + residual as fastbb chunked-fp16 GEMMs
    with fused epilogues (x channels-last fp16 -> the NHWC permutes are free views)."""
    from .gemm import linear
    shortcut = x.permute(0, 2, 3, 1)
    y = self.conv_dw(x).permute(0, 2, 3, 1)
    y = self.norm(y)
    m = self.mlp
    h = linear(y, m.fc1.weight, m.fc1.bias, act="gelu")
    h = m.norm(h)  # V2: GlobalResponseNorm; V1: Identity
    y = linear(h, m.fc2.weight, m.fc2.bias, res=shortcut, gamma=self.gamma)
    return y.permute(0, 3, 1, 2)


def _patch_convnext(model):
    n = 0
    for m in model.modules():
        if type(m).__name__ == "ConvNeXtBlock" and not m.use_conv_mlp and isinstance(m.shortcut, nn.Identity) \
                and isinstance(m.drop_path, nn.Identity) and isinstance(m.mlp.act, nn.GELU) and m.mlp.act.approximate == "none":
            m.forward = _convnext_block_fwd.__get__(m)
            n += 1
    return n


def _fx_fused(model):
    import torch.fx as fx

    from . import fused
    try:
        fused.load()
    except (ImportError, OSError, RuntimeError) as e:
        warnings.warn(f"fastbb fused kernels unavailable ({e}); using PyTorch convolutions", stacklevel=2)
        return model, 0
    try:
        gm = fx.GraphModule(model, fx.Tracer().trace(model))
    except Exception as e:  # noqa: BLE001 - untraceable: keep eager
        print(f"[fastbb.cnn] fx trace failed ({type(e).__name__}), fused convs off")
        return model, 0
    mods = dict(gm.named_modules())
    for n in list(gm.graph.nodes):
        m = mods.get(n.target) if n.op == "call_module" else None
        if isinstance(m, nn.Identity) or (type(m) is nn.BatchNorm2d and getattr(m, "_fastbb_folded", False)):
            n.replace_all_uses_with(n.args[0])  # folded BNs are identities (else fused.replace_bn_act would re-apply them)
            gm.graph.erase_node(n)
    gm.graph.lint()
    gm.recompile()
    n, _ = fused.fuse_all(gm)
    return gm, n


def optimize(model, spec="fold-gacc-fused-trace", sample_hw=(64, 64)):
    flags = set(spec.split("-"))
    if "fused" in flags:
        flags.discard("hacc")  # fused convs replace the cuDNN fp16-acc path
    model = model.eval()
    dev = next(model.parameters()).device
    info = {}
    if "fold" in flags:  # traced + folded where the model is (CPU recommended: no fp32 copy ever reaches the GPU)
        if not any(isinstance(m, nn.BatchNorm2d) for m in model.modules()):
            info["folded_bn"] = 0  # nothing to fold (ConvNeXt, Swin, Hiera, ...): skip the trace
        else:
            try:
                info["folded_bn"] = _fold_bn(model, torch.randn(1, 3, *sample_hw, device=dev))
            except Exception:  # noqa: BLE001 - fixed-size models: trace at their native input size
                hw = tuple(getattr(model, "pretrained_cfg", {}).get("input_size", (3, 224, 224))[-2:])
                info["folded_bn"] = _fold_bn(model, torch.randn(1, 3, *hw, device=dev))
    model.half().cuda()  # fp16 before the move: GPU memory holds only the fp16 weights (exp 008: b1 reserved 202 -> 108 MB)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.weight.dim() == 4:
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
    if "hacc" in flags:
        info["hacc_convs"] = cudnn_conv.patch_convs(model)
        for m in model.modules():  # small convs: cuDNN-frontend call overhead > the 2x MMA rate -> torch conv
            if isinstance(m, nn.Conv2d) and "forward" in m.__dict__:
                kh, kw_ = m.kernel_size  # MACs per INPUT pixel (stride folded in)
                # narrow convs (< 64 in/out channels: EfficientNet / MobileNet stages) are memory-bound: no gain, slower plans
                m._macs_per_px = 0 if min(m.in_channels, m.out_channels) < 64 else \
                    m.in_channels * m.out_channels * kh * kw_ / (m.stride[0] * m.stride[1])
                m.forward = _sized_forward.__get__(m)
    if "fuse" in flags:
        info["fused_blocks"] = _patch_convnext(model)
    if "fused" in flags:  # exp 008 CUTLASS fused convs (bias + residual + act, fp16 acc), fx-rewritten graph
        model, info["fused_convs"] = _fx_fused(model)
    f = FastCNN(model, flags)
    f.info = info
    return f
