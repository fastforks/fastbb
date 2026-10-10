"""DINOv2 ViT-S/B/L/g (+ reg4) as a standalone fastbb feature extractor, loading the OFFICIAL checkpoints
(dl.fbaipublicfiles.com/dinov2/..._pretrain.pth, strict state-dict match; same parameter names as facebookresearch/dinov2).

    from fastbb.models.dinov2 import dinov2
    m = dinov2("vitl14", "weights/dinov2/dinov2_vitl14_pretrain.pth", spec="fh-g0")   # -> FastViT (fp16, CUDA)
    out = m.forward_features(img)           # dict like official: x_norm_clstoken, x_norm_regtokens, x_norm_patchtokens, ...
    feats = m.get_intermediate_layers(img, n=4, reshape=True, return_class_token=True)
img: [B, 3, H, W] ImageNet-normalized, H, W multiples of 14 (any size; nothing compiled per size).
The module tree here only holds parameters (the forward is fastbb.vit.FastViT); it is built directly on the GPU with
empty weights (no CPU random init) and filled from the checkpoint.
"""
import torch
import torch.nn as nn

ARCH = {  # name -> (embed_dim, depth, heads, ffn)
    "vits14": (384, 12, 6, "mlp"), "vitb14": (768, 12, 12, "mlp"), "vitl14": (1024, 24, 16, "mlp"),
    "vitg14": (1536, 40, 24, "swiglu"),
}


class _LS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(d))


class _Mlp(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, h), nn.Linear(h, d)


class _SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w12, self.w3 = nn.Linear(d, 2 * h), nn.Linear(h, d)


class _Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.num_heads, self.scale = heads, (d // heads) ** -0.5
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)


class _Block(nn.Module):
    def __init__(self, d, heads, ffn):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)
        self.attn = _Attn(d, heads)
        self.ls1, self.ls2 = _LS(d), _LS(d)
        self.mlp = _Mlp(d, 4 * d) if ffn == "mlp" else _SwiGLU(d, (int(4 * d * 2 / 3) + 7) // 8 * 8)


class _PatchEmbed(nn.Module):
    def __init__(self, d, ps):
        super().__init__()
        self.proj = nn.Conv2d(3, d, ps, ps)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class DinoV2Params(nn.Module):
    """Parameter container with the official names (patch_embed.proj, cls_token, pos_embed, register_tokens, mask_token,
    blocks.N.{norm1,attn.qkv,attn.proj,ls1.gamma,norm2,mlp.*,ls2.gamma}, norm)."""

    def __init__(self, arch, num_register_tokens=0, img_size=518, patch_size=14):
        super().__init__()
        d, depth, heads, ffn = ARCH[arch]
        self.embed_dim, self.patch_size, self.num_register_tokens = d, patch_size, num_register_tokens
        self.interpolate_antialias = num_register_tokens > 0      # official hub: reg models antialias=True, offset 0
        self.interpolate_offset = 0.0 if num_register_tokens else 0.1
        self.patch_embed = _PatchEmbed(d, patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d))
        self.pos_embed = nn.Parameter(torch.empty(1, (img_size // patch_size) ** 2 + 1, d))
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, d)) if num_register_tokens else None
        self.mask_token = nn.Parameter(torch.empty(1, d))
        self.blocks = nn.ModuleList([_Block(d, heads, ffn) for _ in range(depth)])
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.head = nn.Identity()


def dinov2(arch, weights, spec=None, device="cuda", dtype=torch.float16, graphs=False):
    """Build DINOv2 `arch` (vits14 | vitb14 | vitl14 | vitg14; '_reg' suffix or a *reg4* checkpoint -> 4 registers)."""
    from .. import (
        DEFAULT_VIT,
        install_vit,  # noqa: F401
        parse_vit_spec,
    )
    from ..vit import FastViT
    # weights: mmap on the CPU, cast to fp16 there, module built on the meta device and the tensors assigned -> the GPU
    # only ever holds the fp16 weights (no fp32 copy / no CPU random init); pos_embed stays fp32 (interpolated in fp32)
    sd = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    nreg = sd["register_tokens"].shape[1] if "register_tokens" in sd else 0
    with torch.device("meta"):
        m = DinoV2Params(arch.replace("_reg", ""), num_register_tokens=nreg, img_size=int(round((sd["pos_embed"].shape[1] - 1) ** 0.5)) * 14)
    sd = {k: v.to(torch.float32 if k == "pos_embed" else dtype).to(device) for k, v in sd.items()}
    m.load_state_dict(sd, strict=True, assign=True)
    spec = spec or DEFAULT_VIT
    if spec.endswith("-graph"):
        spec, graphs = spec[:-6], True
    attn, gacc = parse_vit_spec(spec)
    return FastViT(m.eval(), attn=attn, gacc=gacc, dtype=dtype, graphs=graphs)
