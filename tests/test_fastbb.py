from unittest.mock import patch

import pytest
import torch

from fastbb import normconv
from fastbb.cnn import _fx_fused


def test_missing_optional_extension_keeps_pytorch_model():
    model = torch.nn.Sequential(torch.nn.Conv2d(8, 8, 3), torch.nn.ReLU()).eval()
    x = torch.randn(1, 8, 5, 7)
    before = model(x)
    with patch("fastbb.fused.load", side_effect=RuntimeError("no compiler")):
        with pytest.warns(UserWarning, match="using PyTorch"):
            result, count = _fx_fused(model)
    assert result is model and count == 0
    torch.testing.assert_close(result(x), before)


def test_extension_cache_changes_with_torch_abi(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(normconv, "_loaded", [False])
    paths = []
    def exists(path):
        paths.append(path)
        return True
    with patch("fastbb.normconv.os.path.exists", side_effect=exists), patch("torch.ops.load_library"):
        normconv.load()
        normconv._loaded[0] = False
        monkeypatch.setattr(torch, "__version__", "different-build")
        normconv.load()
    assert paths[0] != paths[1]
    assert all(p.startswith(str(tmp_path)) for p in paths)


@pytest.mark.parametrize("base", [0.0, 10.0, 1000.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("affine", [False, True])
def test_instance_norm_constant_is_finite_and_centered(base, dtype, affine):
    from fastbb.instnorm import _instance_norm_nhwc
    m = torch.nn.InstanceNorm2d(8, affine=affine)
    x = torch.full((1, 8, 4, 4), base, dtype=dtype).contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(_instance_norm_nhwc(m, x), torch.zeros_like(x), rtol=0, atol=0)


def test_instance_norm_large_mean_matches_centered_reference():
    from fastbb.instnorm import _instance_norm_nhwc
    x = (1000 + torch.randn(2, 8, 8, 8)).half()
    m = torch.nn.InstanceNorm2d(8, affine=True)
    expected = torch.nn.functional.instance_norm(x.double(), weight=m.weight.double(), bias=m.bias.double(), eps=m.eps).half()
    torch.testing.assert_close(_instance_norm_nhwc(m, x), expected, rtol=0.002, atol=0.002)


@pytest.mark.parametrize("tracked", [True, False])
def test_bn_folding_preserves_branched_consumers(tracked):
    from fastbb.cnn import _fold_bn
    class Branched(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 1)
            self.bn = torch.nn.BatchNorm2d(4, track_running_stats=tracked)
            with torch.no_grad():
                self.bn.weight.fill_(3)
        def forward(self, x):
            y = self.conv(x)
            return y + self.bn(y)
    model = Branched().eval()
    x = torch.randn(2, 3, 4, 5)
    expected = model(x)
    assert _fold_bn(model, x) == 0
    torch.testing.assert_close(model(x), expected)


def test_vit_position_cache_reloads_weights():
    from fastbb.vit import FastViT
    model = torch.nn.Module()
    model.pos_embed = torch.nn.Parameter(torch.zeros(1, 5, 8))
    model.patch_size = 2
    model.blocks = torch.nn.ModuleList()
    fast = FastViT(model)
    assert fast.pos_embed(2, 2).count_nonzero() == 0
    state = model.state_dict()
    state["pos_embed"] = torch.ones_like(state["pos_embed"])
    model.load_state_dict(state)
    torch.testing.assert_close(fast.pos_embed(2, 2), torch.ones_like(model.pos_embed))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deterministic_native_instance_norm():
    from fastbb.instnorm import _instance_norm_kernel
    torch.manual_seed(17)
    module = torch.nn.InstanceNorm2d(64).cuda()
    x = (1000 + torch.randn(1, 64, 128, 192)).cuda().half().contiguous(memory_format=torch.channels_last)
    expected = torch.nn.functional.instance_norm(x.cpu().double(), eps=module.eps).cuda().half()
    first = _instance_norm_kernel(module, x)
    torch.testing.assert_close(first, expected, atol=.002, rtol=.002)
    for _ in range(5):
        assert torch.equal(_instance_norm_kernel(module, x), first)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(1, 8, 64, 48), (2, 16, 32, 40), (1, 64, 96, 312), (1, 8, 300, 400)])
@pytest.mark.parametrize("base", [0.0, 1000.0, 60000.0])
@pytest.mark.parametrize("affine", [False, True])
def test_eager_instance_norm_matches_fp32_reference(shape, base, affine):
    """CUDA eager path (fp32 mean + Welford std, batch-norm kernel apply, no fp32 copy) == the fp32 formula."""
    from fastbb.instnorm import _instance_norm_nhwc, _instance_norm_ref
    torch.manual_seed(5)
    m = torch.nn.InstanceNorm2d(shape[1], affine=affine).cuda()
    if affine:
        torch.nn.init.normal_(m.weight)
        torch.nn.init.normal_(m.bias)
    x = (base + torch.randn(shape) * (1 if base < 1000 else 8)).half().cuda().contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        y = _instance_norm_nhwc(m, x)
        assert y.dtype == x.dtype and y.is_contiguous(memory_format=torch.channels_last)
        torch.testing.assert_close(y, _instance_norm_ref(m, x), atol=4e-3, rtol=4e-3)
        c = torch.full(shape, base, device="cuda", dtype=torch.float16).contiguous(memory_format=torch.channels_last)
        expected = m.bias.detach().half().view(1, -1, 1, 1).expand(shape) if affine else torch.zeros_like(c)
        torch.testing.assert_close(_instance_norm_nhwc(m, c), expected, atol=0, rtol=0)


def test_trace_accepts_assertions_rejects_control_flow():
    from fastbb.trace import Traced

    class Asserting(torch.nn.Module):
        def forward(self, x):
            torch._assert(x.shape[-1] == 6, f"width {x.shape[-1]}")
            return x * 2

    class Branching(torch.nn.Module):
        def forward(self, x):
            if x.sum() > 0:
                return x * 2
            return x - 1

    x = torch.randn(1, 3, 4, 6)
    ok = Traced(Asserting(), shapes=None, enabled=True)
    torch.testing.assert_close(ok(x), x * 2)
    assert ok.info["trace"] == "ok"
    bad = Traced(Branching(), shapes=None, enabled=True)
    bad(x)
    assert bad.info["trace"].startswith("rejected (Python control flow")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_launch_odd_parameter_size():
    """Parameter buffers ending in a 4-byte field (size not a multiple of 8) must launch."""
    from fastbb import kernels as _k
    from fastbb.instnorm import _stats_kernel
    x = torch.randn(1, 16, 8, 8, device="cuda").half().contiguous(memory_format=torch.channels_last)
    st = _stats_kernel(_k.Kernels.get(), x, 1e-5)
    xd = x.double()
    torch.testing.assert_close(st[0, :, 0].double(), xd.mean((0, 2, 3)), atol=1e-4, rtol=1e-4)
