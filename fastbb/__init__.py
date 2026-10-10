"""fastbb: fast drop-in backbones shared by many stereo / mono depth models (exp 006).

    import fastbb
    fastbb.install_vit(model.pretrained, "fh-g0")   # DINOv2-style ViT: get_intermediate_layers -> fast path

Spec strings for install_vit: "<attn>-g<gemm>", attn in sdpa | ff | fh | hh (see attention.py), gemm accumulation
g0 = fp32, g1 = fp16 for all ViT GEMMs, g2 = fp16 except fc2 (K = 4C),
g3 = fastbb chunked-fp16 GEMM kernel (fp16 rate, ~fp32 error) with fused bias/GELU/LayerScale-residual epilogues. "off" leaves the module untouched.
"""
import os

DEFAULT_VIT = os.environ.get("FASTBB_VIT", "fh-g0")


def parse_vit_spec(spec):
    attn, _, g = spec.partition("-")
    gacc = {"g0": False, "g1": True, "g2": "nofc2", "g3": "chunk", "": False}[g]
    return attn, gacc


def install_vit(vit, spec=None, out_dtype=None):
    """Make `vit.get_intermediate_layers` the fastbb path (weights converted to fp16 in place). Returns the FastViT or None."""
    spec = spec or DEFAULT_VIT
    if spec == "off":
        return None
    from .vit import FastViT
    attn, gacc = parse_vit_spec(spec)
    return FastViT(vit, attn=attn, gacc=gacc, out_dtype=out_dtype).install()


def install_vit_blocks(model, spec=None, convert=True):
    """Block-level drop-in for any ViT-style model (timm VisionTransformer, official DINOv2, DAv2 copies): see vit.py."""
    from .vit import install_vit_blocks as _ivb
    return _ivb(model, spec or DEFAULT_VIT, convert)


def optimize_convnet(module, spec="fp16", keep_bf16=(), cast_args=None):
    """Any conv module used at full image sizes (RAFT encoders, DPT heads, decoders): BN folding, InstanceNorm NHWC, dtype +
    channels-last, optional fp16-acc convs ("fp16-hacc"). In place; see fastbb/raft.py optimize_encoder."""
    from .raft import optimize_encoder
    return optimize_encoder(module, spec, keep_bf16, cast_args)
