"""Keep only stereo encoder operators in an already-synced fastbb_vendored directory."""
import ast
import os
import re
import subprocess
import sys
from pathlib import Path


def checked_sub(pattern, replacement, text):
    text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(
            f"Expected exactly one vendor edit for {pattern!r}, got {count}"
        )
    return text

def trim_normconv(text):
    tree = ast.parse(text)
    conv = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "conv"
    )
    first_removed = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "conv_in"
    )
    lines = text.splitlines(keepends=True)
    header = "".join(lines[conv.lineno - 1 : conv.body[0].lineno - 1])
    header = checked_sub(r"\bpre\s*=\s*None\s*,\s*", "", header)
    header = checked_sub(r"\bstats\s*=\s*False\s*,\s*", "", header)
    body = "".join(lines[conv.body[0].lineno - 1 : conv.end_lineno])
    body = checked_sub(
        r"^    if pre is not None:\n        x = in_apply\(x, pre, mode=0\)\n", "", body
    )
    body = checked_sub(r"\bres_mode,\s*stats,\s*cfg\b", "res_mode, False, cfg", body)
    text = (
        "".join(lines[: conv.lineno - 1])
        + header
        + body
        + "".join(lines[conv.end_lineno : first_removed.lineno - 1])
    )
    parsed = ast.parse(text)
    forbidden = {"pre", "stats", "in_apply", "conv_in"}
    if any(
        isinstance(node, ast.Name) and node.id in forbidden for node in ast.walk(parsed)
    ):
        raise ValueError("Removed normalization symbols remain in the RAFT wrapper")
    return text.rstrip() + "\n"


def trim(dest):
    p = dest / 'csrc/norm_conv.cu'
    if p.exists():
        text = p.read_text()
        start, end = text.index('// Restormer GDFN core:'), text.index('TORCH_LIBRARY(fastbb_nc, m)')
        text = text[:start] + text[end:]
        text = '\n'.join(line for line in text.split('\n') if not any(f'"{op}"' in line or f'"{op}(' in line for op in ('dwgate', 'ln_lastdim')))
        p.write_text(text)
    p = dest / 'fastbb_kernels.cu'
    if not p.exists():
        # RAFT uses only the context encoder's convolution/BatchNorm epilogue.
        source = dest / 'csrc/norm_conv.cu'
        text = source.read_text()
        start, end = text.index('// part [G, 2, K]'), text.index('TORCH_LIBRARY(fastbb_nc, m)')
        text = text[:start] + text[end:]
        text = '\n'.join(line for line in text.split('\n') if '"in_stats_final' not in line and '"in_apply' not in line)
        source.write_text(text)
        wrapper = dest / 'normconv.py'
        text = trim_normconv(wrapper.read_text())
        start = text.index('"""'); end = text.index('"""', start + 3) + 3
        text = text[:start] + '"""Vendored CUTLASS convolution/BatchNorm epilogues for the RAFT context encoder."""' + text[end:]
        wrapper.write_text(text)
        return
    text = p.read_text()
    text = '#include <cuda_fp16.h>\n#include <stdint.h>\n\n' + text[text.index('// ------------------------------------------------------------------------------------------------ instance norm NHWC'):]
    p.write_text(text)
    p = dest / 'kernels.py'
    text = p.read_text()
    start, end = text.index('SIGNATURES = {'), text.index('_CTYPE =')
    text = text[:start] + '''SIGNATURES = {
    "in_stats_h": "ppiii", "in_stats_b": "ppiii", "in_stats_final": "ppiiiiifpp",
    "in_apply_h": "ppppppiiifi", "in_apply_b": "ppppppiiifi",
}
DYN_SMEM = {"in_stats_h": 256 * 16 * 4, "in_stats_b": 256 * 16 * 4}
''' + text[end:]
    text, count = re.subn(r"^([ \t]+)from \.trace import mark_invisible\s*\n\s*mark_invisible\(name\)",
                          r'\1raise RuntimeError("raw encoder kernels cannot be traced; export the portable model")', text,
                          flags=re.MULTILINE)
    if count != 2 or "mark_invisible" in text:
        raise ValueError(f"expected the trace guards of launch and launch_packed, replaced {count}")
    p.write_text(text)
    env = dict(os.environ, PYTHONPATH=str(dest.parent))
    subprocess.run([sys.executable, '-m', 'fastbb_vendored.kernels'], env=env, check=True)


if __name__ == '__main__':
    trim(Path(sys.argv[1]).resolve())
