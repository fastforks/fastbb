"""Opt-in CUDA-graph replay for any inference callable (fastbb "max" tier for small inputs, where the per-kernel CPU launch
cost dominates). One graph per (input shapes, dtypes), captured on the first call at a new shape (one warm-up run + capture,
a few ms; nothing compiled or tuned). Outputs are cloned out of the graph's static buffers unless noclone=True.
Not for models with data-dependent control flow / host syncs inside forward.

    from fastbb.graphs import graphed
    f = graphed(model)          # f(x) == model(x)
"""
import torch


def _flat(o, out):
    if torch.is_tensor(o):
        out.append(o)
    elif isinstance(o, (list, tuple)):
        for v in o:
            _flat(v, out)
    elif isinstance(o, dict):
        for v in o.values():
            _flat(v, out)
    return out


def _clone(o):
    if torch.is_tensor(o):
        return o.clone()
    if isinstance(o, (list, tuple)):
        return type(o)(_clone(v) for v in o)
    if isinstance(o, dict):
        return {k: _clone(v) for k, v in o.items()}
    return o


class Graphed:
    def __init__(self, fn, noclone=False, max_graphs=16):
        self.fn, self.noclone, self.max_graphs = fn, noclone, max_graphs
        self._g, self._pool = {}, None

    def __call__(self, *xs):
        key = tuple((tuple(x.shape), x.dtype, x.device, x.stride()) for x in xs)
        g = self._g.get(key)
        if g is None:
            if len(self._g) >= self.max_graphs:  # bound the memory held by static buffers
                self._g.clear()
                self._pool = None
            st = [x.clone() for x in xs]
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.fn(*st)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                out = self.fn(*st)
            self._pool = graph.pool()
            g = self._g[key] = (graph, st, out)
        graph, st, out = g
        for d, x in zip(st, xs):
            d.copy_(x)
        graph.replay()
        return out if self.noclone else _clone(out)


def graphed(fn, noclone=False):
    return Graphed(fn, noclone)
