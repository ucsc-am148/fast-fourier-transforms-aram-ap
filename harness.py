"""Harness for the FFT ladder F1..F7.

This file is GIVEN -- do not edit. It wires together your kernels.py +
twiddles.py modules into prepare / alloc / run functions, and provides the
buffer-cycler used by the F6 / F7 recursion.

Module toggle: set FFT_REF=1 in the environment to swap in kernels_golden
(a stub that serves precomputed (x, y) pairs from golden.pt) and twiddles_ref.
Useful for verifying the harness works end-to-end before you have student
code that runs; NOT a working reference kernel.
"""

import math
import os

import torch
import triton

# ---- module toggle: student modules vs reference modules ----
# FFT_REF=1 swaps in the golden-tensor stub (kernels_golden) -- a precomputed
# (x, y) lookup served through the same module surface as kernels.py. It's a
# smoke-test of the harness wiring, NOT a working reference kernel.
USE_REF = os.environ.get('FFT_REF', '0') == '1'
if USE_REF:
    import kernels_golden as kernels
    import twiddles_ref as twiddles
else:
    import kernels
    import twiddles


# =============================================================================
# Buffer cycler for F6 / F7 recursion
# =============================================================================
# At every cyc.next() the pair about to be handed out has already had its last
# read. Invariant the pipeline relies on:
#   - at most 2 cycler pairs are live at once (this op's input + its output)
#   - each recursion level hands exactly one pair (t1) down to its child, and
#     the child consumes it in its very first op (the T1 transpose)
#   - the top-level input lives in in_re/in_im, which the cycler never returns
# 3 scratch pairs (a, b, c) suffice. Inserting a step between two transposes
# or reading a cycler pair two ops after it was written breaks this -- bump
# the pool (add 'd', 'e') if you reorder _f6_rec / _f7_rec.

class _Cycle:
    def __init__(self, bufs, names):
        self.bufs = bufs
        self.names = names
        self.i = 0

    def next(self):
        n = self.names[self.i % len(self.names)]
        self.i += 1
        return self.bufs[n + '_re'], self.bufs[n + '_im']


# =============================================================================
# F1: DFT-as-matmul (fp16 storage, fp32 accumulators)
# =============================================================================

def f1_prepare(N, device='cuda'):
    W_re, W_im = twiddles.make_dft_matrix(N, dtype=torch.float16, device=device)
    return {'N': N, 'W_re': W_re, 'W_im': W_im}


def f1_alloc(N, B, device='cuda'):
    return {
        'x_re': torch.empty((B, N), dtype=torch.float16, device=device),
        'x_im': torch.empty((B, N), dtype=torch.float16, device=device),
        'y_re': torch.empty((B, N), dtype=torch.float32, device=device),
        'y_im': torch.empty((B, N), dtype=torch.float32, device=device),
    }


def f1_run(x_complex, plan, bufs):
    B, N = x_complex.shape
    assert N == plan['N'], f"N mismatch: x has {N}, plan has {plan['N']}"
    bufs['x_re'].copy_(x_complex.real.to(torch.float16))
    bufs['x_im'].copy_(x_complex.imag.to(torch.float16))
    kernels.f1_launch(bufs['x_re'], bufs['x_im'],
                      plan['W_re'], plan['W_im'],
                      bufs['y_re'], bufs['y_im'])
    return torch.complex(bufs['y_re'], bufs['y_im'])


# =============================================================================
# F2: radix-2 Cooley-Tukey (fp32 throughout)
# =============================================================================

def f2_prepare(N, device='cuda'):
    tw_re, tw_im = twiddles.make_radix2_twiddles(N, dtype=torch.float32, device=device)
    perm = twiddles.bit_reversal_perm(N, device=device)
    return {'N': N, 'tw_re': tw_re, 'tw_im': tw_im, 'perm': perm}


def f2_alloc(N, B, device='cuda'):
    return {
        'x_re': torch.empty((B, N), dtype=torch.float32, device=device),
        'x_im': torch.empty((B, N), dtype=torch.float32, device=device),
        'y_re': torch.empty((B, N), dtype=torch.float32, device=device),
        'y_im': torch.empty((B, N), dtype=torch.float32, device=device),
    }


def f2_run(x_complex, plan, bufs):
    B, N = x_complex.shape
    assert N == plan['N'], f"N mismatch: x has {N}, plan has {plan['N']}"
    bufs['x_re'].copy_(x_complex.real)
    bufs['x_im'].copy_(x_complex.imag)
    kernels.f2_launch(bufs['x_re'], bufs['x_im'], bufs['y_re'], bufs['y_im'],
                      plan['tw_re'], plan['tw_im'], plan['perm'])
    return torch.complex(bufs['y_re'], bufs['y_im'])


# =============================================================================
# F3: Bailey six-step at N = N1 * N2 (fp32 throughout, radix-2 inner)
# =============================================================================

def f3_prepare(N1, N2, device='cuda'):
    tw_re_n1, tw_im_n1 = twiddles.make_radix2_twiddles(N1, dtype=torch.float32, device=device)
    tw_re_n2, tw_im_n2 = twiddles.make_radix2_twiddles(N2, dtype=torch.float32, device=device)
    bt_re, bt_im = twiddles.make_bailey_cross_twiddles(
        N1, N2, N1 * N2, dtype=torch.float32, device=device,
    )
    return {
        'N': N1 * N2, 'N1': N1, 'N2': N2,
        'LOG2_N1': int(math.log2(N1)),
        'LOG2_N2': int(math.log2(N2)),
        'perm_n1': twiddles.bit_reversal_perm(N1, device=device),
        'perm_n2': twiddles.bit_reversal_perm(N2, device=device),
        'tw_re_n1': tw_re_n1, 'tw_im_n1': tw_im_n1,
        'tw_re_n2': tw_re_n2, 'tw_im_n2': tw_im_n2,
        'bt_re': bt_re, 'bt_im': bt_im,
    }


def f3_alloc(N, B, device='cuda'):
    def buf():
        return torch.empty(B * N, dtype=torch.float32, device=device)
    return {
        'in_re': buf(), 'in_im': buf(),
        'mid_re': buf(), 'mid_im': buf(),
        'out_re': buf(), 'out_im': buf(),
    }


def f3_run(x_complex, plan, bufs):
    B, N = x_complex.shape
    assert N == plan['N'], f"N mismatch: x has {N}, plan has {plan['N']}"
    bufs['in_re'].copy_(x_complex.real.contiguous().reshape(-1))
    bufs['in_im'].copy_(x_complex.imag.contiguous().reshape(-1))
    kernels.f3_launch(
        bufs['in_re'], bufs['in_im'],
        bufs['out_re'], bufs['out_im'],
        bufs['mid_re'], bufs['mid_im'],
        plan, B,
    )
    return torch.complex(bufs['out_re'], bufs['out_im']).reshape(B, N)


# =============================================================================
# F4: tcFFT radix-16 (N = 256; fp16 storage / fp32 accumulators)
# =============================================================================

def f4_prepare(N, device='cuda'):
    assert N == 256, f"F4 ships only the L=2 (N=256) path; got {N}"
    F_re, F_im = twiddles.make_dft_matrix(16, dtype=torch.float16, device=device)
    tw_re, tw_im = twiddles.make_radix16_twiddles(N, device=device)
    return {
        'N': N, 'L': 2,
        'F_re': F_re, 'F_im': F_im,
        'tw_re': tw_re, 'tw_im': tw_im,
    }


def f4_alloc(N, B, device='cuda'):
    return {
        'x_re': torch.empty((B, N), dtype=torch.float16, device=device),
        'x_im': torch.empty((B, N), dtype=torch.float16, device=device),
        'y_re': torch.empty((B, N), dtype=torch.float16, device=device),
        'y_im': torch.empty((B, N), dtype=torch.float16, device=device),
    }


def f4_run(x_complex, plan, bufs):
    B, N = x_complex.shape
    assert N == plan['N'], f"N mismatch: x has {N}, plan has {plan['N']}"
    bufs['x_re'].copy_(x_complex.real.to(torch.float16))
    bufs['x_im'].copy_(x_complex.imag.to(torch.float16))
    kernels.f4_kernel_L2[(triton.cdiv(B, kernels.F4_L2_BLOCK_B),)](
        bufs['x_re'], bufs['x_im'], bufs['y_re'], bufs['y_im'],
        plan['F_re'], plan['F_im'],
        plan['tw_re'], plan['tw_im'],
        B, 1,
        BLOCK_B=kernels.F4_L2_BLOCK_B, STAGE_STOP=plan['L'], STORE_T=False,
        num_warps=4, num_stages=1,
    )
    return torch.complex(bufs['y_re'].to(torch.float32),
                         bufs['y_im'].to(torch.float32))


# =============================================================================
# F5: Bailey at N = 65536 with F4 as inner FFT (6-launch pipeline)
# =============================================================================

F5_N1 = 256
F5_N2 = 256
F5_N = F5_N1 * F5_N2


def f5_prepare(device='cuda'):
    bt_re, bt_im = twiddles.make_bailey_cross_twiddles(
        F5_N1, F5_N2, F5_N, dtype=torch.float16, device=device,
    )
    return {
        'N': F5_N, 'N1': F5_N1, 'N2': F5_N2,
        'f4_plan': f4_prepare(256, device=device),
        'bt_re': bt_re, 'bt_im': bt_im,
    }


def f5_alloc(B, device='cuda'):
    def buf():
        return torch.empty((B, F5_N), dtype=torch.float16, device=device)
    return {
        'in_re': buf(), 'in_im': buf(),
        'b0_re': buf(), 'b0_im': buf(),
        'b1_re': buf(), 'b1_im': buf(),
        'b2_re': buf(), 'b2_im': buf(),
    }


def f5_run(x_complex, plan, bufs):
    B, N = x_complex.shape
    assert N == plan['N'], f"N mismatch: x has {N}, plan has {plan['N']}"
    bufs['in_re'].copy_(x_complex.real.to(torch.float16))
    bufs['in_im'].copy_(x_complex.imag.to(torch.float16))
    kernels.f5_launch(
        bufs['in_re'], bufs['in_im'],
        bufs['b0_re'], bufs['b0_im'],
        bufs['b1_re'], bufs['b1_im'],
        bufs['b2_re'], bufs['b2_im'],
        plan, B,
    )
    # final lands in b0
    return torch.complex(bufs['b0_re'].to(torch.float32),
                         bufs['b0_im'].to(torch.float32))


# =============================================================================
# F6 / F7: recursive 2-factor Bailey for all powers of 2
# =============================================================================

def _f67_prepare(N, device='cuda'):
    """Shared plan builder for F6 and F7 -- they use identical plans."""
    chunks = kernels.f6_factor(N)
    plan = {'N': N, 'chunks': chunks, 'device': device}
    plan['f4_plan'] = f4_prepare(256, device=device) if 256 in chunks else None
    plan['dft_mats'] = {
        R: twiddles.make_dft_R_padded(R, device=device)
        for R in set(chunks) if R != 256
    }
    plan['tw'] = []
    for i in range(len(chunks) - 1):
        m0 = chunks[i]
        M = math.prod(chunks[i + 1:])
        Ni = m0 * M
        tw_re, tw_im = twiddles.make_bailey_cross_twiddles(
            m0, M, Ni, dtype=torch.float16, device=device,
        )
        plan['tw'].append((m0, M, Ni, tw_re, tw_im))
    return plan


f6_prepare = _f67_prepare
f7_prepare = _f67_prepare


def _f67_alloc(N, B, device='cuda'):
    def buf():
        return torch.empty(B * N, dtype=torch.float16, device=device)
    return {
        'in_re': buf(), 'in_im': buf(),
        'a_re': buf(), 'a_im': buf(),
        'b_re': buf(), 'b_im': buf(),
        'c_re': buf(), 'c_im': buf(),
    }


f6_alloc = _f67_alloc
f7_alloc = _f67_alloc


def f6_run(x_complex, plan, bufs):
    B, N = x_complex.shape
    assert N == plan['N'], f"N mismatch: x has {N}, plan has {plan['N']}"
    bufs['in_re'].copy_(x_complex.real.to(torch.float16).reshape(-1))
    bufs['in_im'].copy_(x_complex.imag.to(torch.float16).reshape(-1))
    cyc = _Cycle(bufs, ['a', 'b', 'c'])
    out_re, out_im = kernels._f6_rec(bufs['in_re'], bufs['in_im'], B, plan['chunks'], plan, cyc)
    return torch.complex(out_re.to(torch.float32),
                         out_im.to(torch.float32)).reshape(B, N)


def f7_run(x_complex, plan, bufs):
    B, N = x_complex.shape
    assert N == plan['N'], f"N mismatch: x has {N}, plan has {plan['N']}"
    bufs['in_re'].copy_(x_complex.real.to(torch.float16).reshape(-1))
    bufs['in_im'].copy_(x_complex.imag.to(torch.float16).reshape(-1))
    cyc = _Cycle(bufs, ['a', 'b', 'c'])
    out_re, out_im = kernels._f7_rec(bufs['in_re'], bufs['in_im'], B, plan['chunks'], plan, cyc)
    return torch.complex(out_re.to(torch.float32),
                         out_im.to(torch.float32)).reshape(B, N)
