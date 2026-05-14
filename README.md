# FFT kernel ladder — assignment template

A seven-step Triton FFT ladder (F1 → F7). Each kernel removes a limitation of
the previous one. You implement them in order; later kernels reuse earlier
ones.

## The ladder

| # | Algorithm | Size regime | Builds on |
|---|---|---|---|
| F1 | DFT as one dense complex matmul (four `tl.dot`) | small N (16..256), O(N^2) | lecture-11 matmul |
| F2 | radix-2 Cooley-Tukey, one program does the whole transform in registers | medium N, runs out of registers past N ~ 16384 | -- |
| F3 | Bailey six-step FFT, N = N1 * N2: two transposes + two radix-2 sub-FFTs | composite N (N1, N2 each within F2's range) | F2 (with `BAILEY_EPILOGUE` / `STRIDED_STORE` flags) |
| F4 | tcFFT radix-16 single-program, N = 256: 2 stages of (permute + twiddle + length-16 DFT via four `tl.dot`) | N = 256, fp16 storage / fp32 accumulators | F1 idea at radix 16 |
| F5 | Bailey six-step at N1 = N2 = 256 with F4 as inner FFT (6 launches) | N = 65536 | F3's transpose + F4 |
| F6 | Recursive 2-factor Bailey: factor N = 2^k into chunks `[256...] + [16...] + [small]` | all powers of 2 | F3's transpose + F4 + a small padded-DFT kernel |
| F7 | F6 with Scale+T2 and FFT-m0+T3 fused via `STORE_T` flags on F6/F4 kernels | all powers of 2 (bitwise-equal to F6) | F6 kernels (with `STORE_T=True`) |

## Files

```
fft-template/
  README.md           this file
  ALGORITHMS.md       math + pipeline + indexing identities for each rung (READ THIS)
  twiddles.py         host-side twiddle helpers (STUDENT IMPLEMENTS) — 3 canonical patterns
  twiddles_ref.pyc    working reference for twiddles.py (bytecode-only)
  kernels.py          @triton.jit kernels + per-F pipeline drivers (STUDENT IMPLEMENTS)
  kernels_golden.py   stub that serves precomputed reference outputs in FFT_REF=1 mode
  golden.pt           the precomputed outputs (don't edit)
  harness.py          prepare/alloc/run dispatch, buffer cycler, FFT_REF env toggle (GIVEN)
  sanity_check.py     per-F validation vs torch.fft.fft + F7-vs-F6 bitwise check
  requirements.txt    torch>=2.4, triton>=3.6
```

## What you implement

Two files: `twiddles.py` and `kernels.py`. The signatures are the ones the
harness calls — your job is to fill the bodies. When your code passes
`sanity_check.py`, you're done.

### `twiddles.py` — five helpers, three canonical patterns


| pattern | helper | shape | used by |
|---|---|---|---|
| radix-2 length-N/2 | `make_radix2_twiddles` | (N//2,) | F2, F3 |
| per-stage radix-16 | `make_radix16_twiddles` | (L, 16, N//16) fp16 | F4, and via F4 by F5/F6/F7 |
| Bailey cross-term | `make_bailey_cross_twiddles` | (m0, M) | F3, F5, F6, F7 |

Plus two scaffolding tables and a permutation:
- `make_dft_matrix(N)` — (N, N) full DFT matrix for F1.
- `make_dft_R_padded(R)` — (16, 16) DFT for length R in {2,4,8,16} (F6/F7 small chunks).
- `bit_reversal_perm(N)` — (N,) int32 (F2/F3).

All twiddle helpers use the convention `exp(-2*pi*i * ...)` (forward FFT,
negative angle) and return `(re, im)` tuples of separate real-valued tensors.

### `kernels.py` — Triton kernels + pipeline structure

Seven `@triton.jit` kernels:
- `f1_kernel` — DFT as four-`tl.dot` complex matmul.
- `f2_kernel` — radix-2 single-program FFT, with `BAILEY_EPILOGUE` and `STRIDED_STORE` `tl.constexpr` flags so F3 can reuse it (no `f2_a_kernel` / `f2_b_kernel` forks).
- `transpose_kernel` — `(B, R, C) -> (B, C, R)` paired-re/im transpose.
- `f4_kernel_L2` — radix-16 length-256 FFT with `STORE_T` flag (when `True`, writes the transposed (rows_outer, 256, M) layout F7 needs).
- `bailey_scale_kernel` — elementwise `w_N^{n1 kM}` multiply with `STORE_T` flag (when `True`, fuses with a transpose; replaces F7's `scale_transpose_kernel`).
- `dft_kernel` — padded-to-16 small DFT (`R` in {2,4,8,16}) with `STORE_T` flag.

Pipeline drivers:
- `f3_launch` — the 4-step T1 → F2-A → T2 → F2-B sequence.
- `f5_launch` — the 6-step Bailey-with-F4-unmodified sequence.
- `_f6_rec` — recursive 2-factor Bailey split: T1 → recurse → Scale → T2 → FFT-m0 → T3.
- `_f7_rec` — same recursion with Scale+T2 and FFT-m0+T3 each fused (`STORE_T=True`).

The thin launch wrappers (`_transpose`, `_fft_chunk`, `_scale`) are given in
`kernels.py` so you can call them from your pipelines without writing
boilerplate twice.

## What is given (do not edit)

- `ALGORITHMS.md`: the math + pipeline + indexing identities for every rung.
  This is your algorithmic reference — start here before implementing each F.
- `harness.py`: per-F `prepare`/`alloc`/`run` wrappers, the `_Cycle` buffer
  triple-cycler used by `_f6_rec` / `_f7_rec`, and the `FFT_REF` env-var
  toggle.
- `sanity_check.py`: the test runner.
- `kernels_golden.py` + `golden.pt`: in `FFT_REF=1` mode the harness imports
  this module in place of `kernels`. Every kernel call becomes a lookup
  into `golden.pt` (a frozen set of `(x, y)` pairs produced offline). This
  is a smoke-test of your harness wiring and environment — *not* a working
  reference kernel. You can't develop F5/F6/F7 against a "reference F4" by
  setting `FFT_REF=1`; each rung is standalone.
- `twiddles_ref.pyc`: working reference for twiddles, bytecode-only.

## How to run

From `~/am148/fft-template/`:

```bash
pip install -r requirements.txt

# 1. Verify the reference works on your machine (all 7 rows PASS):
FFT_REF=1 python sanity_check.py

# 2. Run your implementation:
python sanity_check.py

# Per-helper twiddle unit tests (run before sanity_check.py while you debug):
python twiddle_check.py
FFT_REF=1 python twiddle_check.py
```

If `sanity_check.py` FAILs and you can't tell whether the kernel or the
twiddle is wrong, run `twiddle_check.py` first — it validates each helper
in `twiddles.py` against hardcoded reference values at small N.

`FFT_REF=1` is keyed by the default `SEED`, `B`, and `N` in
`sanity_check.py`. If you edit any of those, `FFT_REF=1` will raise
`KeyError: no golden entry for ...` — that's expected. To revalidate at a
non-default config, run with `FFT_REF=0` (or unset) so your kernel is
compared against `torch.fft.fft` live.

Pass criteria:
- F1, F2, F3 max-rel-err < 1e-4 vs `torch.fft.fft` (fp32 paths).
- F4, F5, F6, F7 max-rel-err < 1e-2 vs `torch.fft.fft` (fp16 storage; floor grows
  roughly like sqrt(number of stages)).
- F7 must additionally be bitwise-equal to F6 (the fusion preserves bytes).

## Precision contract (why fp16 has a 1e-2 floor)

F2 and F3 are fp32 throughout, so their error floor is ~1e-7.

F4, F5, F6, F7 follow the tcFFT paper's dtype contract: fp16 storage in
registers / HBM, fp32 matmul and twiddle accumulators, fp16 cast between
stages. Error floor is ~1e-3 and grows roughly like sqrt(number of stages).

One real limit, not a bug: a unit-amplitude length-N tone has a spectral peak
of magnitude N, which overflows fp16 (max 65504) for N >~ 65504. Smoke tests
scale the tone by 1/256 to keep the peak in range.
