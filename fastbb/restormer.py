"""Restormer-style TransformerBlocks (MDTA channel attention + GDFN gated depthwise FFN; Restormer, DLNR's
Channel_Attention_Transformer_Extractor, ...) in NHWC fp16, general (any size, no per-size work).

    from fastbb.restormer import optimize
    n = optimize(model.extractor)        # in place: every TransformerBlock(norm1, attn, norm2, ffn) gets the fast forward
    n = optimize(model.extractor, "friendly")   # plain PyTorch ops only (ONNX / TensorRT-exportable), same restructuring

Same math as upstream, fewer passes:
- LayerNorm (WithBias / BiasFree) over the channel dim of the NHWC view (one fused kernel), no rearranges
- MDTA: qkv 1x1 conv as a GEMM on the NHWC view, depthwise 3x3 (fastbb dwconv kernel), per-channel L2 norms of q / k in one
  fp32 reduction, q^T k as one GEMM with fp32 output (cosine = q^T k / (|q| |k|), = F.normalize then matmul), softmax per head;
  attn @ v and project_out folded into ONE per-sample C x C matrix M = blockdiag(attn)^T Wp^T, so x + project_out(attn @ v)
  is a single batched GEMM with the residual as the GEMM accumulator input (baddbmm)
- GDFN: project_in GEMM, depthwise 3x3 + GELU gate in one kernel (fastbb.normconv dwgate), project_out GEMM with the
  residual as accumulator input (addmm)
Blocks keep their modules (weights are read from them; fp16 expected, e.g. after a model.half()).
spec "friendly": F.layer_norm, F.conv2d (groups) depthwise convs + torch GELU gate, F.normalize + fp16 bmm for q^T k (as
upstream), F.linear GEMMs; the attention @ v / project_out fold and the residual-accumulating GEMMs stay.
"""
import torch
import torch.nn.functional as F

from . import normconv
from .gemm import linear as _fb_linear


def _linear(x, w, b):
    """GDFN project_in: fastbb chunked-fp16-accumulate GEMM (prebuilt, error ~ fp32 accumulate) where it applies."""
    return _fb_linear(x, w, b)


def _ln(p, i, xh):
    """LayerNorm over channels of the NHWC tensor (fastbb.normconv ln_lastdim: one read, one write)."""
    w, b, mode = p[f"ln{i}"]
    if p["friendly"]:
        if mode == 0:
            return F.layer_norm(xh, (xh.shape[-1],), w.to(xh.dtype), b.to(xh.dtype), 1e-5)
        var = torch.var(xh.float(), dim=-1, keepdim=True, unbiased=False)
        return (xh * torch.rsqrt(var + 1e-5).to(xh.dtype)) * w.to(xh.dtype)
    return torch.ops.fastbb_nc.ln_lastdim(xh, w, b, 1e-5, mode)


def _w2(conv):
    """[out_channels, in] weight (prepare passes may have zero-padded the outputs to a multiple of 8: sliced off)."""
    o = conv.out_channels
    return conv.weight[:o].reshape(o, -1), (None if conv.bias is None else conv.bias[:o])


def _version(blk):
    return tuple((id(p), None if p.is_inference() else p._version, p.device, p.dtype) for p in blk.parameters())


def _prep(blk):
    at, ff = blk.attn, blk.ffn
    C = at.project_out.out_channels
    heads = at.num_heads
    p = {}
    w, b = _w2(at.qkv)
    p["wqkv"], p["bqkv"] = w.contiguous(), b
    w, b = _w2(at.qkv_dwconv)                                  # [3C, 9]
    p["wdw"] = w.reshape(3 * C, 3, 3).permute(1, 2, 0).contiguous()   # fastbb dwconv weight layout: [R, S, C]
    p["wdw4"] = w.reshape(3 * C, 1, 3, 3).contiguous()                # F.conv2d (friendly)
    p["bdw"] = b
    w, b = _w2(at.project_out)
    p["wpT"] = w.t().float().reshape(heads, C // heads, C).contiguous()   # [h, c, C]
    p["bp"] = b
    p["temp"] = at.temperature.float().reshape(1, heads, 1, 1)
    # GDFN with the hidden width padded to a multiple of 8 (zero weights: padded channels are gelu(0) * 0 = 0) so that all
    # GEMMs / vector loads are aligned: layout [a | 0 | b | 0]
    win, bin_ = _w2(ff.project_in)
    h = win.shape[0] // 2
    hp = (h + 7) // 8 * 8
    z = lambda t, n: torch.cat([t, t.new_zeros((n,) + tuple(t.shape[1:]))]) if n else t
    p["win"] = torch.cat([z(win[:h], hp - h), z(win[h:], hp - h)]).contiguous()
    p["bin"] = None if bin_ is None else torch.cat([z(bin_[:h], hp - h), z(bin_[h:], hp - h)]).contiguous()
    wg, bg = _w2(ff.dwconv)                                    # [2h, 9]
    wg = wg.float()
    p["wg"] = torch.cat([z(wg[:h], hp - h), z(wg[h:], hp - h)]).t().contiguous()   # [9, 2hp]
    p["bg"] = None if bg is None else torch.cat([z(bg[:h].float(), hp - h), z(bg[h:].float(), hp - h)]).contiguous()
    p["wg4"] = p["wg"].t().reshape(2 * hp, 1, 3, 3).to(win.dtype).contiguous()       # F.conv2d (friendly)
    p["bg4"] = None if p["bg"] is None else p["bg"].to(win.dtype)
    wout, bout = _w2(ff.project_out)                           # [C, h]
    p["woutT"] = z(wout.t(), hp - h).contiguous()              # [hp, C]
    p["bout"] = bout
    p["C"], p["heads"] = C, heads
    for i, nm in ((1, blk.norm1), (2, blk.norm2)):
        body = nm.body
        if hasattr(body, "bias"):   # WithBias_LayerNorm: (x - mu) / sqrt(var + 1e-5) * w + b
            p[f"ln{i}"] = (body.weight.float().contiguous(), body.bias.float().contiguous(), 0)
        else:                       # BiasFree_LayerNorm: x / sqrt(var + 1e-5) * w
            p[f"ln{i}"] = (body.weight.float().contiguous(), None, 1)
    p["version"] = _version(blk)
    return p


CHUNK = 1024


def _gram(q, k):
    """q^T k over the HW rows ([B, HW, C] strided views) with fp32 output, as a batched GEMM over 1024-row chunks + sum
    (one long-K GEMM per sample leaves the GPU mostly idle)."""
    B, HW, C = q.shape
    S = HW // CHUNK
    out = []
    for b in range(B):
        qb, kb = q[b], k[b]
        g = None
        if S > 1:
            qs = qb[:S * CHUNK].unflatten(0, (S, CHUNK))
            ks = kb[:S * CHUNK].unflatten(0, (S, CHUNK))
            g = torch.bmm(qs.transpose(1, 2), ks, out_dtype=torch.float32).sum(0)
        r0 = S * CHUNK if S > 1 else 0
        if r0 < HW:
            gr = torch.mm(qb[r0:].t(), kb[r0:], out_dtype=torch.float32)
            g = gr if g is None else g + gr
        out.append(g)
    return torch.stack(out)


def _attn(blk, p, x3, xh):
    """x3: residual [B, HW, C]; xh: [B, H, W, C] -> x3 + project_out(attn(norm1(x)))"""
    B, H, W, C = xh.shape
    heads, c = p["heads"], C // p["heads"]
    y = _ln(p, 1, xh)
    qkv = F.linear(y, p["wqkv"], p["bqkv"])                                       # [B, H, W, 3C]
    if p["friendly"]:
        qkv = F.conv2d(qkv.permute(0, 3, 1, 2), p["wdw4"], p["bdw"], 1, 1, 1, 3 * C)
        t = qkv.permute(0, 2, 3, 1).reshape(B, H * W, 3 * C)
        q, k, v = t[:, :, :C], t[:, :, C:2 * C], t[:, :, 2 * C:]
        qn, kn = F.normalize(q, dim=1), F.normalize(k, dim=1)                     # as upstream (over HW)
        g = torch.bmm(qn.transpose(1, 2), kn).float()
        nrm = None
    else:
        qkv = torch.ops.fastbb.dwconv(qkv.permute(0, 3, 1, 2), p["wdw"], p["bdw"], [1, 1], [1, 1], 0)
        t = qkv.permute(0, 2, 3, 1).reshape(B, H * W, 3 * C)                      # contiguous view
        nrm = torch.linalg.vector_norm(t[:, :, :2 * C], dim=1, dtype=torch.float32).clamp_(min=1e-12)  # [B, 2C]
        q, k, v = t[:, :, :C], t[:, :, C:2 * C], t[:, :, 2 * C:]
        g = _gram(q, k)                                                           # [B, C, C] fp32
    if heads > 1:  # per-head diagonal blocks
        g = torch.diagonal(g.view(B, heads, c, heads, c), dim1=1, dim2=3).permute(0, 3, 1, 2)
    else:
        g = g[:, None]
    if nrm is not None:
        g = g / (nrm[:, :C].reshape(B, heads, c, 1) * nrm[:, C:].reshape(B, heads, 1, c))
    attn = torch.softmax(g * p["temp"], dim=-1)                                   # [B, h, c, c] fp32
    M = torch.matmul(attn.transpose(-1, -2), p["wpT"]).reshape(B, C, C).to(x3.dtype)  # fold attn @ v + project_out
    out = torch.baddbmm(x3, v, M)
    if p["bp"] is not None:
        out += p["bp"]
    return out


def _ffn(blk, p, x3, xh):
    B, H, W, C = xh.shape
    y = _ln(p, 2, xh)
    if p["friendly"]:
        z = F.linear(y, p["win"], p["bin"])
        d = F.conv2d(z.permute(0, 3, 1, 2), p["wg4"], p["bg4"], 1, 1, 1, z.shape[-1]).permute(0, 2, 3, 1)
        a, b = d.chunk(2, dim=-1)
        g = (F.gelu(a) * b).contiguous()
    else:
        z = _linear(y, p["win"], p["bin"])                                         # [B, H, W, 2h]
        g = torch.ops.fastbb_nc.dwgate(z, p["wg"], p["bg"])                        # [B, H, W, h]
    out = torch.addmm(x3.reshape(-1, C), g.reshape(-1, g.shape[-1]), p["woutT"])
    if p["bout"] is not None:
        out += p["bout"]
    return out.reshape(B, H * W, C)


def _block_forward(self, x):
    p = self._fastbb_p
    version = _version(self)
    if version != p["version"] or any(v[1] is None for v in version):
        friendly = p["friendly"]
        p = self._fastbb_p = _prep(self)
        p["friendly"] = friendly
    if x.dtype != torch.float16 or x.shape[1] % 8:
        return self._fastbb_orig(x)
    B, C, H, W = x.shape
    xh = x.permute(0, 2, 3, 1).contiguous()                                        # no-op for channels-last input
    x3 = xh.reshape(B, H * W, C)
    x3 = _attn(self, p, x3, xh)
    x3 = _ffn(self, p, x3, x3.reshape(B, H, W, C))
    return x3.reshape(B, H, W, C).permute(0, 3, 1, 2)                              # channels-last NCHW view


def _is_block(m):
    return all(hasattr(m, a) for a in ("norm1", "attn", "norm2", "ffn")) and hasattr(m.attn, "qkv_dwconv") and \
        hasattr(m.attn, "temperature") and hasattr(m.ffn, "project_in") and hasattr(m.ffn, "dwconv")


@torch.no_grad()
def optimize(model, spec="max"):
    """In place; returns the number of blocks patched. spec: "max" (fastbb kernels) | "friendly" (plain PyTorch)."""
    friendly = spec == "friendly"
    if not friendly:
        import fastbb.fused as ff
        ff.load()
        normconv.load()
    n = 0
    for m in model.modules():
        if _is_block(m) and "_fastbb_p" not in m.__dict__:
            m._fastbb_p = _prep(m)
            m._fastbb_p["friendly"] = friendly
            m._fastbb_orig = m.forward
            m.forward = _block_forward.__get__(m)
            n += 1
    return n
