# Algorithms — F1..F7

Math + pipeline + indexing identities for each rung of the FFT ladder. No
Triton, no kernel structure. If you can re-derive the small-N spot-checks
in `twiddle_check.py` from the formulas here, you understand the algorithm.

For dtype / precision conventions (fp32 throughout for F1..F3, fp16 storage
with fp32 accumulators for F4..F7), see the README's "Precision contract"
section. That's a kernel concern; the algorithms here are dtype-agnostic.

## 0. Notation

- `N` is the transform length. `B` is the batch dimension.
- `w_N = exp(-2*pi*i / N)` is the primitive N-th root of unity (forward-FFT
  sign convention; negative angle).
- `x[n]` is the input signal, `X[k]` is its DFT. Both are 0-indexed.
- The DFT is `X[k] = sum_{n=0}^{N-1} x[n] * w_N^{n*k}` for `k` in `[0, N)`.
- Index decompositions follow the convention that linear indices are written
  low-axis-fast, e.g. for `N = N1 * N2` we write `n = n2*N1 + n1` (so `n1` is
  the fast axis in the linear input). Each rung's section restates its
  convention.
- Complex tensors are stored as `(re, im)` pairs of real-valued tensors
  throughout (see `twiddles.py`); the formulas below treat them as a single
  complex value.

## 1. F1 — Dense DFT

```
X[k] = sum_{n=0}^{N-1} x[n] * w_N^{n*k}
```

Equivalently, with the `(N, N)` DFT matrix `W[j, k] = w_N^{j*k}`,
this is the matrix product `Y = X @ W^T`, where `X` is the `(B, N)` batch
of input rows and `Y` is the `(B, N)` batch of output rows.

That's it — F1 is the DFT computed as one dense complex matmul, no
factorization. Cost is O(N^2) per signal. Used as the baseline against
which every other rung's correctness can ultimately be checked, and as a
small-N building block for the F4 inner radix-16 DFT (`make_dft_matrix`).

## 2. F2 — Radix-2 Cooley-Tukey, in registers

Bit-reverse the input, then run `log2(N)` butterfly stages.

**Bit-reversal.** For `N = 2^L`, `rev[i]` is the integer whose `L`-bit
binary representation is `i`'s bits in reversed order. The kernel loads
`v[j] = x[rev[j]]`, then operates on `v` in place.

**Butterfly stage `s` (s = 0, 1, ..., L-1).** Partition `[0, N)` into pairs
`(j, j XOR 2^s)`. For each pair, replace

```
v_new[j]         = v[j] + w * v[j XOR 2^s]
v_new[j XOR 2^s] = v[j] - w * v[j XOR 2^s]
```

The twiddle `w` depends only on the low `s` bits of `j`:

```
w = w_N^{ (j & (2^s - 1)) * (N >> (s + 1)) }
```

i.e. the twiddle table at index `(j & (2^s - 1)) * (N >> (s + 1))` in a
length-`N/2` table holding `w_N^0, w_N^1, ..., w_N^{N/2 - 1}`. The kernel
reads that table; `make_radix2_twiddles` builds it.

**Why one program does the whole transform.** F2's design choice is that
each program holds the entire length-`N` signal in registers across all
`log2(N)` stages. This avoids any HBM round-trip between stages, but the
total register footprint grows with `N`: past `N ~ 16384` the kernel spills
and performance collapses. F3 exists to break that wall.

## 3. F3 — Bailey six-step at N = N1 * N2

The Bailey decomposition lets a length-`N` FFT be expressed as two batches
of shorter FFTs (length `N2` and length `N1`) joined by a single
elementwise multiply.

**Index decomposition.** Write `n = n2*N1 + n1` for the input and
`k = k1*N2 + k2` for the output, with `n1 ∈ [0, N1)`, `n2 ∈ [0, N2)`,
`k1 ∈ [0, N1)`, `k2 ∈ [0, N2)`. Substituting into the DFT sum and
simplifying mod `N`:

```
X[k1*N2 + k2]
  = sum_{n1, n2} x[n2*N1 + n1]
                 * w_{N2}^{n2*k2}     (inner FFT over n2, length N2)
                 * w_N^{n1*k2}        (Bailey cross-twiddle)
                 * w_{N1}^{n1*k1}     (outer FFT over n1, length N1)
```

**Pipeline.** Logical view of the `(B, N)` input as `(B, N2, N1)`:

| step | op | input shape | output shape |
|---|---|---|---|
| T1 | transpose last two axes | `(B, N2, N1)` | `(B, N1, N2)` |
| F2-A | length-`N2` FFT along last axis, *then* multiply by `w_N^{n1*k2}` | `(B, N1, N2)` | `(B, N1, N2)` |
| T2 | transpose last two axes | `(B, N1, N2)` | `(B, N2, N1)` |
| F2-B | length-`N1` FFT along last axis, *then* strided store as `(B, N1, N2)` | `(B, N2, N1)` | `(B, N1, N2)` |

After F2-B the linear output index is `b*N + k1*N2 + k2`, matching the
output decomposition above.

**Two transposes are absorbed.** A naive six-step Bailey would be: T1,
F2-A, Scale, T2, F2-B, T3. F3 saves two kernels by (a) fusing the Bailey
multiply into F2-A's epilogue (the `BAILEY_EPILOGUE` constexpr flag) and
(b) having F2-B store with stride `N2` so it writes the post-T3 layout
directly (the `STRIDED_STORE` constexpr flag). The algorithmic content is
unchanged — only kernel launches are saved.

## 4. F4 — tcFFT radix-16 at N = 256, L = 2

Radix-16 Cooley-Tukey instead of radix-2. With `N = 16^L` and `L = 2`, the
algorithm is `L` stages of (permute + per-stage twiddle multiply +
length-16 DFT).

**Tile view.** Reshape `(B, 256)` to `(B, 16, 16)`. Each input element
`x[n]` decomposes into `L` base-16 digits `n = d_0 * 16^(L-1) + d_1 * 16^(L-2)
+ ... + d_{L-1}` (high digit first). The `(B, 16, ..., 16)` tile is indexed
by `(d_0, d_1, ..., d_{L-1})`.

**Stage `s`** (s = 0, 1, ..., L-1) is three sub-steps:

1. **Permute** the tile so axis `s` is at position 0. This brings the
   not-yet-transformed digit `d_s` to the front.
2. **Per-stage twiddle multiply** (skipped at `s = 0`).
3. **Length-16 DFT** along position 0, transforming digit `d_s` into output
   digit `e_{L-1-s}`.

After all `L` stages, the tile is indexed by output digits
`(e_{L-1}, e_{L-2}, ..., e_0)` (output digit ordering is reversed from
input — this is the analogue of radix-2's bit reversal). One final permute
reorders to natural output layout.

**Per-stage twiddle formula.** The twiddle at stage `s` reads

```
tw[m, c] = exp(-2*pi*i * m * t / 16^(s+1))
```

where `m` is the row index (the digit being transformed at this stage) and
`t` is an integer reconstructed from already-transformed output digits whose
labels appear at axis position > 0 in the current tile layout:

```
t = sum_{j=0}^{s-1} value-of-digit-e_{L-1-j}-at-column-index-c * 16^j
```

The bookkeeping for which axis carries which digit at each stage is handled
by `_column_axis_labeling(L)` (see `twiddles.py:47-65`); the per-stage
permute schedule is `(s,) + (axes 0..L-1 except s, in original order)`. At
`s = 0` there are no preceding output digits to read, so `t = 0` and the
twiddle is identically 1 (the kernel skips the multiply).

**Length-16 DFT** is `_cdot(tile, F)` where `F` is the `(16, 16)` DFT matrix
from `make_dft_matrix(16)`.

## 5. F5 — Bailey at N = 65536 with F4 as inner FFT

Same Bailey factorization as F3, with `N1 = N2 = 256` and the inner FFT
replaced by F4. Because F4 is left unmodified (no `BAILEY_EPILOGUE`, no
`STRIDED_STORE`), F5 runs the full six-step pipeline:

| step | op | input shape | output shape |
|---|---|---|---|
| T1 | transpose last two axes | `(B, 256, 256)` | `(B, 256, 256)` |
| F4-A | length-256 FFT along last axis | `(B, 256, 256)` | `(B, 256, 256)` |
| Scale | elementwise multiply by `w_N^{n1*k2}` | `(B, 256, 256)` | `(B, 256, 256)` |
| T2 | transpose last two axes | `(B, 256, 256)` | `(B, 256, 256)` |
| F4-B | length-256 FFT along last axis | `(B, 256, 256)` | `(B, 256, 256)` |
| T3 | transpose last two axes | `(B, 256, 256)` | `(B, 256, 256)` |

The index identities are the F3 ones with `N1 = N2 = 256`. The Bailey
cross-twiddle `w_N^{n1*k2}` is now an `(256, 256)` fp16 table (vs F3's fp32);
that's a precision choice, not algorithmic.

F5 is the warm-up step for F6/F7: it demonstrates that F4 composes through
Bailey at one level of nesting. F6 then recurses.

## 6. F6 — Recursive 2-factor Bailey for all powers of 2

Factor `N = 2^k` into a list of chunk sizes (innermost-first) and apply the
2-factor Bailey identity recursively. The Bailey factorization from F3 is
used unchanged; what changes is that the *inner* length-`M` FFT can itself
be another Bailey transform.

**Chunk recipe** (`f6_factor(N)`). Prefer 256-length chunks (handled by
F4), then 16-length (handled by a padded length-16 DFT), then one small
leftover in `{2, 4, 8}` for any remaining bits. Examples:

| `N` | `chunks` (innermost-first) |
|---|---|
| 256 | `[256]` |
| 1024 | `[16, 4]` |
| 4096 | `[256, 16]` |
| 65536 | `[256, 256]` |
| 1048576 | `[256, 256, 16]` |

The leftmost chunk (innermost) is the fastest input axis.

**Recursion.** Let the current level have chunks `[m_0, m_1, ..., m_{p-1}]`
and define `M = prod(chunks[1:])`, `N_i = m_0 * M` (the transform length at
this level). If `p = 1` (leaf), apply one length-`m_0` FFT and return.
Otherwise:

| step | op | input shape | output shape |
|---|---|---|---|
| T1 | transpose | `(rows, M, m_0)` | `(rows, m_0, M)` |
| recurse | length-`M` FFT on `chunks[1:]` | `(rows*m_0, M)` | `(rows*m_0, M)` |
| Scale | multiply by `w_{N_i}^{n_1 * k_M}` | `(rows, m_0, M)` | `(rows, m_0, M)` |
| T2 | transpose | `(rows, m_0, M)` | `(rows, M, m_0)` |
| FFT-m_0 | length-`m_0` FFT along last axis | `(rows, M, m_0)` | `(rows, M, m_0)` |
| T3 | transpose | `(rows, M, m_0)` | `(rows, m_0, M)` |

The Bailey cross-twiddle at this level is `w_{N_i}^{n_1 * k_M}` with shape
`(m_0, M)`; the harness precomputes one such table per recursion level.

The length-`m_0` FFT kernel is chosen by chunk size: `m_0 = 256` uses F4;
`m_0 in {2, 4, 8, 16}` uses `dft_kernel` (a length-`R` DFT padded to a
`(16, 16)` tl.dot so the matmul-instruction shape requirement still holds).

## 7. F7 — F6 with Scale+T2 and FFT-m_0+T3 fused

Same recursion as F6. Two pairs of adjacent ops at each non-leaf level
fuse into single kernels because they share an axis layout:

- **Scale + T2.** Scale is an elementwise multiply over `(rows, m_0, M)`;
  T2 permutes that into `(rows, M, m_0)`. A single kernel can multiply
  and write transposed in one pass, eliminating the intermediate scaled
  buffer that would otherwise sit between the two ops.
- **FFT-m_0 + T3.** The inner FFT operates on the `m_0` axis of
  `(rows, M, m_0)`; T3 permutes the result into `(rows, m_0, M)` for the
  caller. A single FFT kernel can write its output transposed.

Implementation pointer: both fusions are wired via `STORE_T=True` on
`bailey_scale_kernel` and on the FFT kernels (`f4_kernel_L2`, `dft_kernel`)
respectively. The kernel logic that *computes* the result is unchanged;
only the address arithmetic at store time differs.

**Bitwise equality contract.** F7 must produce byte-identical output to
F6, not just within the F6 tolerance. Because the fusion changes only the
store layout — not the order of scalar arithmetic — every floating-point
op runs in the same order across F6 and F7, so the bit-pattern of the
output matches exactly. Any reordering of arithmetic that produces a
different scalar accumulation pattern (e.g. swapping the order of two
adds inside the FFT, or folding the Bailey multiply into the matmul
accumulator) would break the contract even if F7 still passes the F7
tolerance check independently.
