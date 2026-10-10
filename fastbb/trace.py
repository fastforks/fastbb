"""fastbb.trace (opt-in): graph-free launch-overhead removal. A TorchScript trace of a function / module, run by the C++
executor with all fusers off (plain aten / custom-op dispatch, nothing compiled or specialized per input size), validated
against eager at the first call. Origin: exp 009 seg_fast/trace.py (pattern from exp 008 clf_fast.Fast); generalized to
several tensor inputs.

    net = fastbb.trace.Traced(fn_or_module, name="mynet")
    y = net(x)            # or net(left, right): positional tensor args only
    # 1st call: eager warm-up, trace, validate on fresh random inputs at the call's shapes AND at a second set of shapes
    # (different H and W); later calls: the trace, or eager for good on any mismatch / exception (printed once).

Rules (exp 008 / 009 lessons):
- every custom kernel must be a torch op (torch.ops.<lib>.*): raw ctypes / cuDNN-frontend launches are invisible to the
  tracer. Such code paths should call fastbb.trace.mark_invisible("what") -> the trace is rejected instead of trusting
  the validation (a replay would read uninitialized memory).
- shape-dependent numbers belong inside the C++ ops (output sizes, scales, tile choices) or must be derived from traced
  tensor sizes; int arguments of custom ops are frozen into the trace (pass sizes as tensors). Python-side caches are
  filled by the eager warm-up before tracing and are NOT re-consulted by the replay (they must be shape-independent).
- data-dependent Python control flow (early exits, .item() decisions) is frozen as well: the validation on random inputs
  only catches it if the random inputs take a different branch -> do not trace such code.
- validation inputs: FRESH random tensors (traced run first, stale allocator memory cannot fake a match); CPU / CUDA RNG
  forked and re-seeded so random draws inside the net match; the whole build runs on forked RNG states (consumes no draws).
- global flags the trace cannot record (fp16-accumulate GEMM flag, monkey patches) must be set by the caller around the call.

Tracer warnings (tensor -> Python bool / number while tracing) mean Python control flow or constants were frozen into the trace;
the trace is then rejected, except for warnings raised by assertions (`assert`, `_assert`, `torch._assert`, e.g. timm's input-size
checks), which cannot change the computation (FASTBB_TRACE_WARNINGS=reject rejects those too, =allow accepts every warning).
Env: FASTBB_TRACE=0 disables (eager), FASTBB_TRACE_TOL (default 1e-2, relative to max |eager|), FASTBB_TRACE_OPT=1 keeps
the profiling graph executor (default off: run the graph as traced). Nothing here changes behaviour of other fastbb modules.
"""
import os
import warnings

import torch

ENABLED = os.environ.get("FASTBB_TRACE", "1") == "1"
TOL = float(os.environ.get("FASTBB_TRACE_TOL", "1e-2"))

_INVISIBLE = []


def mark_invisible(what):
    """call from code paths a trace cannot record (ctypes launches, cuDNN-frontend executions)."""
    if torch.jit.is_tracing():
        _INVISIBLE.append(what)


def jit_plain():
    """TorchScript executor without fusers / codegen: plain aten dispatch from C++ (process-global settings)."""
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    if os.environ.get("FASTBB_TRACE_OPT", "0") != "1":
        torch._C._set_graph_executor_optimize(False)


def _flat(y):
    if isinstance(y, torch.Tensor):
        return [y]
    if isinstance(y, (list, tuple)):
        return [t for v in y for t in _flat(v)]
    if isinstance(y, dict):
        return [t for v in y.values() for t in _flat(v)]
    return []


def alt_shape(shape, mult=32):
    """a second validation shape for one [N, C, H, W] (or [..., H, W]) tensor: different H and W, multiples of `mult`."""
    *lead, h, w = shape
    h2 = max(mult, (h // 2) // mult * mult)
    w2 = max(mult, (w * 3 // 4) // mult * mult)
    if h2 == h:
        h2 = h + mult
    if w2 == w:
        w2 = w + 2 * mult
    return (*lead, h2, w2)


def alt_shapes(shapes, mult=32):
    """default for several inputs: all inputs whose trailing (H, W) equal the first input's get the same new (H, W);
    others (e.g. 1-D parameters) keep their shape."""
    h, w = shapes[0][-2:]
    h2, w2 = alt_shape(shapes[0], mult)[-2:]
    return [(*s[:-2], h2, w2) if len(s) >= 2 and tuple(s[-2:]) == (h, w) else tuple(s) for s in shapes]


def same(r, t, tol, frac=0.0):
    """outputs match: every float element within tol * max|r| (frac > 0: at most frac of the elements outside, for nets
    with nondeterministic kernels (atomics) whose iterations amplify rounding at a few chaotic pixels; sizes frozen into a
    trace still fail: shape mismatch or errors everywhere)."""
    fr, ft = _flat(r), _flat(t)
    if len(fr) != len(ft):
        return False, "n_out"
    for a, b in zip(fr, ft):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
        if a.is_floating_point():
            a, b = a.float(), b.float()
            if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                return False, "nonfinite output"
            lim = tol * (float(a.abs().max()) + 1e-3) if a.numel() else 0.0
            if frac > 0 and a.numel():
                bad = float(((a - b).abs() > lim).float().mean())
                if not bad <= frac:
                    return False, f"{100 * bad:.3g} % of elements off by > {lim:.3g}"
                continue
            d = float((a - b).abs().max()) if a.numel() else 0.0
            if not d <= lim:
                return False, f"max diff {d:.3g}"
        elif not bool((a == b).all()):
            return False, "int mismatch"
    return True, ""


def _like(x, shape, gen):
    """fresh input like x at `shape`: gen(shape, x) if given, else randn (memory format of x kept)."""
    if gen is not None:
        return gen(tuple(shape), x)
    if x.is_floating_point():
        xi = torch.randn(shape, device=x.device, dtype=torch.float32).to(x.dtype)
    else:
        xi = torch.randint(0, 2, shape, device=x.device).to(x.dtype)
    if x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous():
        xi = xi.contiguous(memory_format=torch.channels_last)
    return xi


WARNINGS = os.environ.get("FASTBB_TRACE_WARNINGS", "asserts")  # asserts | reject | allow
_TORCH_DIR = os.path.dirname(torch.__file__) + os.sep
_ASSERT_CALLS = {"_assert", "torch._assert"}
_stmt_cache = {}


def _is_assert(filename, lineno):
    """True if the statement at filename:lineno is an assertion (`assert ...`, `_assert(...)`, `torch._assert(...)`).
    A tensor -> bool / number conversion there only decides whether to raise: it cannot change the traced computation."""
    key = (filename, lineno)
    if key not in _stmt_cache:
        import ast
        import linecache
        res = False
        try:
            tree = ast.parse("".join(linecache.getlines(filename)))
            best = None
            for node in ast.walk(tree):
                if isinstance(node, ast.stmt) and node.lineno <= lineno <= (node.end_lineno or node.lineno):
                    if best is None or (node.end_lineno - node.lineno) <= (best.end_lineno - best.lineno):
                        best = node
            if isinstance(best, ast.Assert):
                res = True
            elif isinstance(best, ast.Expr) and isinstance(best.value, ast.Call):
                res = ast.unparse(best.value.func) in _ASSERT_CALLS
        except (SyntaxError, ValueError, TypeError):
            res = False
        _stmt_cache[key] = res
    return _stmt_cache[key]


class _TracerWarnings:
    """Collects the TracerWarnings of a torch.jit.trace call with the model-code location that caused each one."""

    def __enter__(self):
        import traceback
        self.sites = []
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("always", torch.jit.TracerWarning)
        prev = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            if not issubclass(category, torch.jit.TracerWarning):
                return prev(message, category, filename, lineno, file, line)
            frames = [f for f in traceback.extract_stack()[:-1]
                      if not f.filename.startswith(_TORCH_DIR) and f.filename != __file__ and "warnings" not in f.filename]
            f = frames[-1] if frames else None
            self.sites.append((f.filename, f.lineno) if f else (filename, lineno))
        warnings.showwarning = show
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

    def unsafe(self):
        """Locations of warnings the policy does not accept (FASTBB_TRACE_WARNINGS: asserts = only assertions are accepted,
        reject = none, allow = all)."""
        if WARNINGS == "allow":
            return []
        out = []
        for fn, ln in self.sites:
            if WARNINGS == "reject" or not _is_assert(fn, ln):
                s = f"{os.path.basename(fn)}:{ln}"
                if s not in out:
                    out.append(s)
        return out


class Traced:
    """fn_or_module(*tensors) -> tensor / tuple / list / dict of tensors.
    shapes: callable(list of input shapes) -> list of second-validation shapes (None: only the call's shapes);
    gen: optional callable(shape, like_tensor) -> fresh validation input (default randn; e.g. uniform 0..255 images).
    frac: > 0 = allow that fraction of output elements outside tol (nondeterministic kernels, see same()).
    key: optional callable(list of input shapes) -> hashable size-regime signature. Python branches that depend on the size
         regime (e.g. 'window >= feature map' paths) are frozen into a trace: one trace is built per key, and the second
         validation shape set is picked with the same key (candidates: shapes(...), then growing H / W).
    """

    def __init__(self, fn, name="", shapes=alt_shapes, tol=None, enabled=None, gen=None, key=None, frac=0.0):
        self.module, self.name, self.shapes, self.gen, self.key, self.frac = fn, name, shapes, gen, key, frac
        self.trs = {}
        self.tol = TOL if tol is None else tol
        self.enabled = ENABLED if enabled is None else enabled
        self.tr = None           # None: not built yet; False: rejected (eager); else the traced function
        self.info = {"trace": "pending" if self.enabled else "off"}

    def _alt(self, shp0):
        if not self.shapes:
            return None
        cands = [list(self.shapes(shp0))]
        if self.key is not None:
            h, w = shp0[0][-2:]
            for dh, dw in ((64, 96), (32, 160), (160, 32)):
                cands.append([(*s[:-2], s[-2] + dh, s[-1] + dw) if len(s) >= 2 and tuple(s[-2:]) == (h, w) else tuple(s)
                              for s in shp0])
            cands = [c for c in cands if self.key(c) == self.key(shp0)]
        return cands[0] if cands else None

    def _devs(self, xs):
        return sorted({x.device for x in xs if x.is_cuda}, key=str)

    def _run_seeded(self, f, xs, seed):
        with torch.random.fork_rng(devices=self._devs(xs)):
            torch.manual_seed(seed)
            return f(*xs)

    def build(self, *xs):
        with torch.random.fork_rng(devices=self._devs(xs)):
            self._build(xs)

    def _build(self, xs):
        jit_plain()
        with torch.no_grad():
            self.module(*xs)                                     # eager warm-up: Python-side caches, plans
            del _INVISIBLE[:]
            tr = None
            try:
                mod = self.module
                if isinstance(mod, torch.nn.Module):            # trace a function, not the module (no recursive
                    mod.requires_grad_(False)                   # ScriptModule wrapping; weights become constants)

                    def fn(*t):
                        return mod(*t)
                else:
                    fn = mod
                with _TracerWarnings() as tw:
                    tr = torch.jit.trace(fn, tuple(xs), check_trace=False, strict=False)
                bad = tw.unsafe()
                ok, why = not bad, ("Python control flow or constants were captured at " + ", ".join(bad[:3])) if bad else ""
                if _INVISIBLE:
                    ok, why = False, "invisible ops: " + ",".join(sorted(set(_INVISIBLE)))
                shp0 = [tuple(x.shape) for x in xs]
                alt = self._alt(shp0)
                sets = [shp0] + ([alt] if alt is not None else [])
                for k, shp in enumerate(sets):
                    if not ok:
                        break
                    xi = [_like(x, s, self.gen) for x, s in zip(xs, shp)]
                    t = self._run_seeded(tr, xi, 1234 + k)
                    r = self._run_seeded(self.module, xi, 1234 + k)
                    ok, why = same(r, t, self.tol, self.frac)
                    if not ok:
                        why = f"shapes {shp}: {why}"
            except Exception as e:  # noqa: BLE001 - untraceable Python -> eager
                ok, why = False, f"trace failed: {type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
        self.tr = tr if ok else False
        self.info = {"trace": "ok" if ok else f"rejected ({why})"}
        if not ok:
            print(f"[fastbb.trace] {self.name}: trace rejected -> eager ({why})")

    def __call__(self, *xs):
        if not self.enabled:
            return self.module(*xs)
        if self.key is not None:
            k = self.key([tuple(x.shape) for x in xs])
            if k not in self.trs:
                self.tr = None
                self.build(*xs)
                self.trs[k] = self.tr
            return (self.trs[k] or self.module)(*xs)
        if self.tr is None:
            self.build(*xs)
        return (self.tr or self.module)(*xs)
