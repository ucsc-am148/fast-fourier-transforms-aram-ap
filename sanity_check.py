"""Local sanity check for the FFT template.

Runs each of F1..F7 at a fixed validation size, compares against torch.fft.fft
(fp64 reference for fp16-storage kernels), and reports max relative error +
rough kernel time + PASS/FAIL.

This is NOT the autograder. It verifies correctness; the autograder will run
more sizes and may enforce a per-kernel performance floor.

Usage:
    python sanity_check.py          # run your kernels.py + twiddles.py
    FFT_REF=1 python sanity_check.py  # serve precomputed reference outputs (smoke-test only)

Until you implement each kernel, that row will FAIL with NotImplementedError.
"""

import os
import time
import traceback

import torch

import harness


# Validation sizes. F4 ships only L=2 (N=256). F5 ships only N=65536.
CASES = [
    ('F1', 'f1',   {'N': 64}),
    ('F2', 'f2',   {'N': 1024}),
    ('F3', 'f3',   {'N1': 64, 'N2': 64}),
    ('F4', 'f4',   {'N': 256}),
    ('F5', 'f5',   {}),
    ('F6', 'f6',   {'N': 32768}),
    ('F7', 'f7',   {'N': 32768}),
]
# F2/F3 are fp32 throughout: tight tolerance.
# F1/F4/F5/F6/F7 are fp16-storage: error grows ~sqrt(stages), 1e-2 floor.
TOL_FP32 = 1e-4
TOL_FP16 = 2e-2
TOL_BY_NAME = {'F1': TOL_FP16, 'F2': TOL_FP32, 'F3': TOL_FP32,
               'F4': TOL_FP16, 'F5': TOL_FP16, 'F6': TOL_FP16, 'F7': TOL_FP16}
# Batch sizes per F. F1 needs B >= 16 (tl.dot tile floor).
B_BY_NAME = {'F1': 64, 'F2': 8, 'F3': 8, 'F4': 8,
             'F5': 4, 'F6': 4, 'F7': 4}
SEED = 0


def _max_rel(y, y_ref):
    return ((y - y_ref).abs().max() / y_ref.abs().max()).item()


def _build(prefix, kwargs, B, device):
    """Call f{N}_prepare(...) and f{N}_alloc(...) from harness; return (plan, bufs, N)."""
    prepare = getattr(harness, prefix + '_prepare')
    alloc = getattr(harness, prefix + '_alloc')
    if prefix == 'f5':
        plan = prepare(device=device)
        bufs = alloc(B, device=device)
        N = plan['N']
    elif prefix == 'f3':
        plan = prepare(kwargs['N1'], kwargs['N2'], device=device)
        N = plan['N']
        bufs = alloc(N, B, device=device)
    else:
        N = kwargs['N']
        plan = prepare(N, device=device)
        bufs = alloc(N, B, device=device)
    return plan, bufs, N


def _run(prefix, plan, bufs, x):
    run = getattr(harness, prefix + '_run')
    return run(x, plan, bufs)


def _bench(fn, warmup=2, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def run_one(name, prefix, kwargs, device):
    B = B_BY_NAME[name]
    torch.manual_seed(SEED)
    tol = TOL_BY_NAME[name]
    try:
        plan, bufs, N = _build(prefix, kwargs, B, device)
    except NotImplementedError as e:
        print(f"  {name:>3}  {'-':>9}  {'-':>4}  {'-':>12}  {'-':>8}   "
              f"FAIL  (prepare/alloc NotImplemented: {e})")
        return
    except Exception as e:
        print(f"  {name:>3}  {'-':>9}  {'-':>4}  {'-':>12}  {'-':>8}   "
              f"FAIL  ({type(e).__name__}: {e})")
        return

    x = torch.randn(B, N, dtype=torch.complex64, device=device)
    try:
        y = _run(prefix, plan, bufs, x)
    except NotImplementedError as e:
        print(f"  {name:>3}  {N:>9d}  {B:>4d}  {'-':>12}  {'-':>8}   "
              f"FAIL  (run NotImplemented: {e})")
        return
    except Exception as e:
        print(f"  {name:>3}  {N:>9d}  {B:>4d}  {'-':>12}  {'-':>8}   "
              f"FAIL  ({type(e).__name__}: {e})")
        return

    y_ref = torch.fft.fft(x.to(torch.complex128), dim=1).to(torch.complex64)
    err = _max_rel(y, y_ref)

    try:
        ms = _bench(lambda: _run(prefix, plan, bufs, x))
    except Exception:
        ms = float('nan')

    verdict = "PASS" if err < tol else "FAIL"
    print(f"  {name:>3}  {N:>9d}  {B:>4d}  {err:>12.3e}  {ms:>8.3f}   {verdict}  (tol {tol:.0e})")


def run_f4_stage1_check(device):
    """Run f4_kernel_L2 with STAGE_STOP=1 against a numpy reference.

    With STAGE_STOP=1 the kernel runs only the s=0 path (no twiddles, the
    multiply is skipped). This is equivalent to: view input as (B, 16, 16),
    DFT along axis 1, reshape to (B, 256). Isolates "permute/reshape/output-
    permute is right" from "stage-1 twiddle table is right".
    """
    import triton
    N = 256
    B = 8
    torch.manual_seed(SEED)
    try:
        plan = harness.f4_prepare(N, device=device)
        bufs = harness.f4_alloc(N, B, device=device)
    except NotImplementedError as e:
        print(f"  s=1  {'-':>9}  {'-':>4}  {'-':>12}  {'-':>8}   "
              f"FAIL  (prepare/alloc NotImplemented: {e})")
        return
    except Exception as e:
        print(f"  s=1  {'-':>9}  {'-':>4}  {'-':>12}  {'-':>8}   "
              f"FAIL  ({type(e).__name__}: {e})")
        return

    x = torch.randn(B, N, dtype=torch.complex64, device=device)
    bufs['x_re'].copy_(x.real.to(torch.float16))
    bufs['x_im'].copy_(x.imag.to(torch.float16))
    K = harness.kernels
    try:
        K.f4_kernel_L2[(triton.cdiv(B, K.F4_L2_BLOCK_B),)](
            bufs['x_re'], bufs['x_im'], bufs['y_re'], bufs['y_im'],
            plan['F_re'], plan['F_im'], plan['tw_re'], plan['tw_im'],
            B, 1,
            BLOCK_B=K.F4_L2_BLOCK_B, STAGE_STOP=1, STORE_T=False,
            num_warps=4, num_stages=1,
        )
    except NotImplementedError as e:
        print(f"  s=1  {N:>9d}  {B:>4d}  {'-':>12}  {'-':>8}   "
              f"FAIL  (kernel NotImplemented: {e})")
        return
    except Exception as e:
        print(f"  s=1  {N:>9d}  {B:>4d}  {'-':>12}  {'-':>8}   "
              f"FAIL  ({type(e).__name__}: {e})")
        return

    y = torch.complex(bufs['y_re'].to(torch.float32),
                      bufs['y_im'].to(torch.float32))
    # Reference: length-16 DFT along axis 1 of (B, 16, 16) view.
    y_ref = torch.fft.fft(
        x.view(B, 16, 16).to(torch.complex128), dim=1,
    ).reshape(B, N).to(torch.complex64)
    err = _max_rel(y, y_ref)
    tol = TOL_FP16
    verdict = "PASS" if err < tol else "FAIL"
    print(f"  s=1  {N:>9d}  {B:>4d}  {err:>12.3e}  {'-':>8}   "
          f"{verdict}  (tol {tol:.0e})")


def run_f7_equals_f6(device):
    """F7 must produce bitwise-identical output to F6 (the fusion preserves bytes)."""
    N = 32768
    B = 4
    torch.manual_seed(SEED)
    try:
        plan6, bufs6, _ = _build('f6', {'N': N}, B, device)
        plan7, bufs7, _ = _build('f7', {'N': N}, B, device)
    except (NotImplementedError, Exception) as e:
        print(f"  F7==F6:  could not build plans ({type(e).__name__}: {e})")
        return
    x = torch.randn(B, N, dtype=torch.complex64, device=device)
    try:
        y6 = _run('f6', plan6, bufs6, x)
        y7 = _run('f7', plan7, bufs7, x)
    except (NotImplementedError, Exception) as e:
        print(f"  F7==F6:  could not run ({type(e).__name__}: {e})")
        return
    if torch.equal(y6, y7):
        print(f"  F7==F6:  bitwise-equal at N={N}, B={B}  [PASS]")
    else:
        diff = _max_rel(y7, y6)
        print(f"  F7==F6:  NOT bitwise-equal at N={N}, B={B}; max rel diff={diff:.3e}  [FAIL]")


def main():
    device = 'cuda'
    if not torch.cuda.is_available():
        print("CUDA not available; cannot run FFT kernels.")
        return

    print(f"device:  {torch.cuda.get_device_name(0)}")
    print(f"torch:   {torch.__version__}")
    try:
        import triton
        print(f"triton:  {triton.__version__}")
    except ImportError:
        print("triton:  not installed")
    print(f"FFT_REF: {'1 (reference modules)' if harness.USE_REF else '0 (student modules)'}")
    print()

    print(f"  {'F':>3}  {'N':>9}  {'B':>4}  {'max rel err':>12}  {'ms':>8}   {'verdict':<6}")
    print(f"  {'-'*3}  {'-'*9}  {'-'*4}  {'-'*12}  {'-'*8}   {'-'*6}")
    for name, prefix, kwargs in CASES:
        if name == 'F4':
            run_f4_stage1_check(device)
        run_one(name, prefix, kwargs, device)
    print()
    run_f7_equals_f6(device)


if __name__ == '__main__':
    main()
