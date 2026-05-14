"""Stubbed reference kernels for FFT_REF=1 smoke-tests.

When harness.py imports this module in place of `kernels`, every kernel and
pipeline driver becomes a lookup into `golden.pt` -- a frozen set of (x, y)
pairs produced by the real reference kernels offline. This lets students
verify their environment / harness wiring end-to-end before they have any
working code of their own, WITHOUT shipping the reference kernel source.

What this is NOT: a working reference implementation. The kernels here do
not compute anything; they copy precomputed bytes into the output buffer.
You cannot develop F5/F6/F7 pipelines against a "reference F4" by enabling
FFT_REF=1 here -- each rung is standalone now. Use FFT_REF=1 only as a
"does my environment work" smoke-test.

Loaded once at import time from `golden.pt` (alongside this file). If the
file is missing the module still imports, but every kernel call will raise
a clear FileNotFoundError on first use.

Keys in golden.pt:
    ('F1', B, N)        full F1 run
    ('F2', B, N)        full F2 run
    ('F3', B, N)        full F3 run
    ('F4_S1', B, 256)   f4_kernel_L2 with STAGE_STOP=1 (sanity stage-1 check)
    ('F4', B, 256)      f4_kernel_L2 with STAGE_STOP=L (full F4 via f4_run)
    ('F5', B, 65536)    full F5 run
    ('F6', B, N)        full F6 run (also served for F7 -> bitwise equality)
"""

import math
import os
import warnings

import torch


F4_L2_BLOCK_B = 2
DFT_BLOCK_B = 16
SCALE_BLOCK = 32
TRANSPOSE_BLOCK = 32


_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_PATH = os.path.join(_HERE, 'golden.pt')

try:
    _GOLDENS = torch.load(_GOLDEN_PATH, map_location='cpu', weights_only=False)
except FileNotFoundError:
    _GOLDENS = None


def f6_factor(N: int) -> list[int]:
    """Factor N = 2^k into FFT chunks (recipe published in README)."""
    assert N >= 2 and (N & (N - 1)) == 0, f"N must be a power of 2 >= 2; got {N}"
    k = N.bit_length() - 1
    n256, rb = divmod(k, 8)
    n16, rb2 = divmod(rb, 4)
    rsmall = 1 << rb2
    chunks = [256] * n256 + [16] * n16 + ([rsmall] if rsmall > 1 else [])
    assert math.prod(chunks) == N
    return chunks


f7_factor = f6_factor


def _lookup(key):
    if _GOLDENS is None:
        raise FileNotFoundError(
            f"{_GOLDEN_PATH} not found. FFT_REF=1 mode needs the precomputed "
            f"golden bundle. Either run with FFT_REF=0 (your own kernels) or "
            f"obtain golden.pt from the course materials."
        )
    if key not in _GOLDENS:
        have = sorted(_GOLDENS.keys())
        raise KeyError(
            f"no golden entry for {key}. "
            f"FFT_REF=1 only covers the default sanity_check configs "
            f"({have}). If you've changed SEED, B, or N in sanity_check.py, "
            f"that's expected -- run with FFT_REF=0 instead and compare "
            f"against torch.fft.fft directly."
        )
    return _GOLDENS[key]


def _check_x(key, x_re, x_im, B, N):
    """Sanity-warn if the caller's input doesn't match the stored x.

    Non-fatal: students may experiment with input generation. We just want a
    visible signal that 'FFT_REF=1 PASS' here doesn't mean their kernel
    received the same input the golden was computed on.
    """
    entry = _GOLDENS[key]
    stored_x = entry['x']  # complex64 (B, N) on cpu
    obs_re = x_re.view(B, N).to(torch.float32).cpu()
    obs_im = x_im.view(B, N).to(torch.float32).cpu()
    diff_re = (obs_re - stored_x.real.to(torch.float32)).abs().max().item()
    diff_im = (obs_im - stored_x.imag.to(torch.float32)).abs().max().item()
    # fp16-storage Fs may show small diffs from the complex64 -> fp16 cast in
    # the caller; allow a generous slack.
    if max(diff_re, diff_im) > 1e-2:
        warnings.warn(
            f"FFT_REF=1: caller's input for {key} differs from the stored x "
            f"by {max(diff_re, diff_im):.2e}. The served y was computed on a "
            f"different input; PASS here does not mean your kernel is right.",
            stacklevel=3,
        )


def _serve(key, y_re_out, y_im_out, B, N):
    """Copy golden y[key] into the (re, im) output buffer pair."""
    entry = _lookup(key)
    y = entry['y']  # complex64 on cpu
    device = y_re_out.device
    y_re_out.view(B, N).copy_(y.real.to(y_re_out.dtype).to(device))
    y_im_out.view(B, N).copy_(y.imag.to(y_im_out.dtype).to(device))


def f1_launch(x_re, x_im, W_re, W_im, y_re, y_im):
    B, N = x_re.shape
    key = ('F1', B, N)
    _lookup(key)
    _check_x(key, x_re, x_im, B, N)
    _serve(key, y_re, y_im, B, N)


def f2_launch(x_re, x_im, y_re, y_im, tw_re, tw_im, perm):
    B, N = x_re.shape
    key = ('F2', B, N)
    _lookup(key)
    _check_x(key, x_re, x_im, B, N)
    _serve(key, y_re, y_im, B, N)


def f3_launch(in_re, in_im, out_re, out_im, mid_re, mid_im, plan, B):
    N = plan['N']
    key = ('F3', B, N)
    _lookup(key)
    _check_x(key, in_re, in_im, B, N)
    _serve(key, out_re, out_im, B, N)


def f5_launch(in_re, in_im, b0_re, b0_im, b1_re, b1_im, b2_re, b2_im, plan, B):
    N = plan['N']
    key = ('F5', B, N)
    _lookup(key)
    _check_x(key, in_re, in_im, B, N)
    _serve(key, b0_re, b0_im, B, N)


def _f6_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Top-level: ignore recursion, serve a single golden. Both _f6_rec and
    _f7_rec call into the same key so the F7==F6 bitwise check holds.
    """
    N = math.prod(chunks)
    B = rows
    key = ('F6', B, N)
    _lookup(key)
    _check_x(key, cur_re, cur_im, B, N)
    out_re, out_im = cyc.next()
    _serve(key, out_re, out_im, B, N)
    return out_re, out_im


def _f7_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Serves the same key as _f6_rec -> bitwise-equal output."""
    return _f6_rec(cur_re, cur_im, rows, chunks, plan, cyc)


class _F4ShimKernel:
    """Mimics the Triton-kernel `[grid](...)` calling convention.

    `f4_kernel_L2[grid](args..., BLOCK_B=..., STAGE_STOP=..., STORE_T=..., ...)`
    -> indexing returns self; calling looks up the right golden by STAGE_STOP.
    """

    def __getitem__(self, grid):
        return self

    def __call__(self, x_re, x_im, y_re, y_im, F_re, F_im, tw_re, tw_im,
                 B, M, *,
                 BLOCK_B, STAGE_STOP, STORE_T,
                 num_warps=4, num_stages=1):
        # sanity_check exercises STAGE_STOP in {1, 2}, STORE_T=False, M=1.
        N = 256
        key = ('F4_S1' if STAGE_STOP == 1 else 'F4', B, N)
        _lookup(key)
        _check_x(key, x_re, x_im, B, N)
        _serve(key, y_re, y_im, B, N)


f4_kernel_L2 = _F4ShimKernel()
