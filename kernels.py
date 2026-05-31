"""STUDENT FILE: implement the Triton kernels and pipeline drivers.

You implement:
  - Six @triton.jit kernels: f1_kernel, f2_kernel, transpose_kernel,
    f4_kernel_L2, dft_kernel, bailey_scale_kernel.
  - The f1_launch and f2_launch grid-choice wrappers around them.
  - The pipeline drivers: f3_launch, f5_launch, _f6_rec, _f7_rec.
  - f6_factor: the chunk-recipe for F6/F7.

You do NOT implement (left given below):
  - The thin launch wrappers _transpose, _fft_chunk, _scale, _lookup_tw.
    These are mechanical "pick the grid and launch one kernel" helpers.
  - The tuning constants F4_L2_BLOCK_B, DFT_BLOCK_B, SCALE_BLOCK,
    TRANSPOSE_BLOCK.

The signatures below are the ones the harness calls -- your job is to fill
the bodies. When your code passes sanity_check.py, you're done.
"""

import math

import torch
import triton
import triton.language as tl
from triton import next_power_of_2

# Tunings -- GIVEN.
F4_L2_BLOCK_B = 2
DFT_BLOCK_B = 16
SCALE_BLOCK = 32
TRANSPOSE_BLOCK = 32


# =============================================================================
# Device-function helper: complex matmul
# =============================================================================
# Implement this once -- f1_kernel, f4_kernel_L2, and dft_kernel all call it.


@triton.jit
def _cdot(a_re, a_im, b_re, b_im):
    """Complex matmul Y = A @ B as four real tl.dot calls.

    Returns (y_re, y_im) in fp32 (out_dtype=tl.float32). Caller is responsible
    for any fp16 down-cast on store. Works at any matmul shape tl.dot accepts.

    Used by f1_kernel, f4_kernel_L2, and dft_kernel. Don't reimplement the
    four-tl.dot expansion at each call site -- implement once here, call
    everywhere.
    """

    y_re = tl.dot(a_re, b_re, out_dtype=tl.float32) - tl.dot(a_im, b_im, out_dtype=tl.float32)
    y_im = tl.dot(a_re, b_im, out_dtype=tl.float32) + tl.dot(a_im, b_re, out_dtype=tl.float32)

    return y_re, y_im


# =============================================================================
# Chunk factorization for F6 / F7
# =============================================================================

def f6_factor(N: int) -> list[int]:
    """Factor N = 2^k into FFT chunks.

    Recipe: prefer 256-length chunks (radix-256, handled by f4_kernel_L2), then
    16-length (handled by dft_kernel via the padded radix-16 path), then a
    small leftover in {2, 4, 8} for the remaining bits. chunks[0] is the
    innermost (fastest) input axis. Examples:
        256 -> [256]                4096 -> [256, 16]
        65536 -> [256, 256]         1048576 -> [256, 256, 16]
        64 -> [16, 4]               2 -> [2]
    """
    k = int(round(math.log2(N)))
    assert 2 ** k == N, f"N must be a power of 2, got {N}"
    chunks: list[int] = []
    while k >= 8:
        chunks.append(256)
        k -= 8
    while k >= 4:
        chunks.append(16)
        k -= 4
    if k > 0:
        chunks.append(2 ** k)
    return chunks


f7_factor = f6_factor   # F7 reuses F6's chunk recipe


# =============================================================================
# F1: DFT as one dense complex matmul (four tl.dot)
# =============================================================================

@triton.jit
def f1_kernel(
    x_re_ptr, x_im_ptr,    # (B, N) fp16
    W_re_ptr, W_im_ptr,    # (N, N) fp16; W[n, k]
    y_re_ptr, y_im_ptr,    # (B, N) fp32
    B,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Y = X @ W^T as four (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) tl.dot calls.

    Y[b, n] = sum_k X[b, k] * W[n, k]. Load W in transposed access
    (W_T[k, n] = W[n, k]) so tl.dot reads it the way it wants.

    Use `_cdot(x_re, x_im, W_T_re, W_T_im)` for the per-block complex matmul;
    accumulate its fp32 output into `acc_re` / `acc_im`.

    Dtype contract (same as F4): loads are fp16, `tl.dot` runs with
    `out_dtype=tl.float32` (handled by `_cdot`), accumulator is fp32, store
    is fp32. Allocations in `f1_alloc` already match this -- x_re/x_im are
    fp16, y_re/y_im are fp32.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Get offsets/mask for M and N
    offs_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offs_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    mask_m = offs_m < B
    mask_n = offs_n < N

    # Initialize our accumulators
    acc_re = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, N, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < N

        # X block: (BLOCK_M, BLOCK_K), row-major (B, N)
        x_off = offs_m[:, None] * N + offs_k[None, :]
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_re = tl.load(x_re_ptr + x_off, mask=x_mask, other=0.0)
        x_im = tl.load(x_im_ptr + x_off, mask=x_mask, other=0.0)

        # W^T block: shape (BLOCK_K, BLOCK_N) read from W[n, k] with n on cols
        wt_off = offs_k[:, None] + offs_n[None, :] * N
        wt_mask = mask_k[:, None] & mask_n[None, :]
        wt_re = tl.load(W_re_ptr + wt_off, mask=wt_mask, other=0.0)
        wt_im = tl.load(W_im_ptr + wt_off, mask=wt_mask, other=0.0)

        blk_re, blk_im = _cdot(x_re, x_im, wt_re, wt_im)
        acc_re += blk_re
        acc_im += blk_im

    y_off = offs_m[:, None] * N + offs_n[None, :]
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_re_ptr + y_off, acc_re, mask=y_mask)
    tl.store(y_im_ptr + y_off, acc_im, mask=y_mask)



def f1_launch(x_re, x_im, W_re, W_im, y_re, y_im):
    """Grid: (cdiv(B, BLOCK_M), cdiv(N, BLOCK_N)). One program tiles a
    (BLOCK_M, BLOCK_N) output square. tl.dot needs all three dims >=16, so B
    should be >= 16.
    """
    B, N = x_re.shape
    BLOCK_M = 16
    # BLOCK_M = min(64, next_power_of_2(B))
    BLOCK_K = N
    BLOCK_N = N
    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(N, BLOCK_N))
    f1_kernel[grid](
        x_re, x_im,
        W_re, W_im,
        y_re, y_im,
        B,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        # num_warps=4,
        # num_stages=2,
    )


# =============================================================================
# F2: radix-2 Cooley-Tukey, single program per signal
# =============================================================================
# F3 reuses this kernel! For F2, only BAILEY_EPILOGUE=False, STRIDED_STORE=False need to be implemented.
#
# Call-site cheatsheet:
#   F2 vanilla:  pid -> one signal in (B, N). Grid: (B,).
#                BAILEY_EPILOGUE=False, STRIDED_STORE=False.
#                OUTER_DIM and N_TOTAL unused (pass 1 / 0).
#                bt_*_ptr: pass tw_*_ptr again (sentinel; never read).
#   F2-A (F3):   pid -> (b, n1). Grid: (B*N1,). FFT length N=N2.
#                BAILEY_EPILOGUE=True, STRIDED_STORE=False.
#                OUTER_DIM=N1 (n1 = pid % N1).
#                bt_*_ptr: real Bailey twiddles shape (N1, N2).
#   F2-B (F3):   pid -> (b, k2). Grid: (B*N2,). FFT length N=N1.
#                BAILEY_EPILOGUE=False, STRIDED_STORE=True.
#                OUTER_DIM=N2, N_TOTAL=N1*N2.
#                bt_*_ptr: sentinel.

@triton.jit
def f2_kernel(
    x_re_ptr, x_im_ptr,        # (B, N) fp32 input
    y_re_ptr, y_im_ptr,        # (B, N) fp32 output (layout depends on STRIDED_STORE)
    tw_re_ptr, tw_im_ptr,      # (N/2,) fp32 radix-2 twiddles
    perm_ptr,                   # (N,) int32 bit-reversal index
    bt_re_ptr, bt_im_ptr,       # (OUTER_DIM, N) fp32 Bailey twiddles (BAILEY_EPILOGUE only)
    OUTER_DIM, N_TOTAL,
    N: tl.constexpr,
    LOG2_N: tl.constexpr,
    BAILEY_EPILOGUE: tl.constexpr,
    STRIDED_STORE: tl.constexpr,
):
    """Radix-2 Cooley-Tukey FFT in registers, with optional Bailey epilogue and
    strided store. log2(N) butterfly stages via tl.gather for partner shuffle.
    """

    pid = tl.program_id(0)
    idx = tl.arange(0, N)

    # Bit-reversal load: v[j] = x[rev[j]]
    rev = tl.load(perm_ptr + idx)
    base = pid * N
    v_re = tl.load(x_re_ptr + base + rev)
    v_im = tl.load(x_im_ptr + base + rev)

    # log2(N) butterfly stages
    for s in tl.static_range(LOG2_N):
        bit = 1 << s

        # Twiddle: w[j] = w_N^{(j & (bit-1)) * (N >> (s+1))}
        tw_idx = (idx & (bit - 1)) * (N >> (s + 1))
        w_re = tl.load(tw_re_ptr + tw_idx)
        w_im = tl.load(tw_im_ptr + tw_idx)

        # Partner-pair indices: bit s cleared / set
        low_idx = idx ^ (idx & bit)
        high_idx = low_idx + bit
        v_re_lo = tl.gather(v_re, low_idx, axis=0)
        v_im_lo = tl.gather(v_im, low_idx, axis=0)
        v_re_hi = tl.gather(v_re, high_idx, axis=0)
        v_im_hi = tl.gather(v_im, high_idx, axis=0)

        # t = w * v_high (complex)
        t_re = w_re * v_re_hi - w_im * v_im_hi
        t_im = w_re * v_im_hi + w_im * v_re_hi

        # v_new[j] = v_low + sign * t, sign = +1 (low) / -1 (high)
        sign = tl.where((idx & bit) != 0, -1.0, 1.0)
        v_re = v_re_lo + sign * t_re
        v_im = v_im_lo + sign * t_im

    # Output address: row-major or strided (post-T3 layout for F2-B).
    if STRIDED_STORE:
        b_outer = pid // OUTER_DIM
        rem = pid % OUTER_DIM
        out_off = b_outer * N_TOTAL + idx * OUTER_DIM + rem
    else:
        out_off = base + idx

    tl.store(y_re_ptr + out_off, v_re)
    tl.store(y_im_ptr + out_off, v_im)


def f2_launch(x_re, x_im, y_re, y_im, tw_re, tw_im, perm):
    """
    Grid: (B,). One program per length-N signal. Vanilla mode.
    """

    B, N = x_re.shape
    LOG2_N = int(math.log2(N))
    grid = (B,)
    f2_kernel[grid](
        x_re, x_im,
        y_re, y_im,
        tw_re, tw_im,
        perm,
        tw_re, tw_im,
        1, 0,
        N=N,
        LOG2_N=LOG2_N,
        BAILEY_EPILOGUE=False,
        STRIDED_STORE=False,
        # num_warps=4,
        # num_stages=2,
    )


# =============================================================================
# transpose_kernel: (B, R, C) -> (B, C, R), paired re/im
# =============================================================================

@triton.jit
def transpose_kernel(
    x_re_ptr, x_im_ptr,     # (B*R*C,) fp16 or fp32 input
    y_re_ptr, y_im_ptr,     # (B*R*C,) fp16 or fp32 output
    R, C,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Logical (B, R, C) -> (B, C, R) transpose. Grid: (cdiv(R, BLOCK_R),
    cdiv(C, BLOCK_C), B). Each program copies a (BLOCK_R, BLOCK_C) tile.
    """

    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_r[:, None] < R) & (offs_c[None, :] < C)

    base = pid_b * R * C

    # get input offset
    in_off = base + offs_r[:, None] * C + offs_c[None, :]
    x_re = tl.load(x_re_ptr + in_off, mask=mask)
    x_im = tl.load(x_im_ptr + in_off, mask=mask)

    # store input as transposed
    out_off = base + offs_c[None, :] * R + offs_r[:, None]
    tl.store(y_re_ptr + out_off, x_re, mask=mask)
    tl.store(y_im_ptr + out_off, x_im, mask=mask)


# =============================================================================
# F4: tcFFT radix-16 single-program FFT (N = 256, L = 2)
# =============================================================================
# See the kernel docstring for the tl.permute tuple-literal gotcha.

@triton.jit
def f4_kernel_L2(
    x_re_ptr, x_im_ptr,    # (B, 256) fp16
    y_re_ptr, y_im_ptr,    # (B, 256) or (B//M, 256, M) fp16
    F_re_ptr, F_im_ptr,    # (16, 16) fp16 -- F_16 DFT matrix
    tw_re_ptr, tw_im_ptr,  # (L=2, 16, 16) fp16 stacked stage twiddles
    B, M,
    BLOCK_B: tl.constexpr,
    STAGE_STOP: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """tcFFT length-256 FFT as two stages of (permute + per-stage twiddle +
    length-16 DFT via four tl.dot). fp16 storage, fp32 matmul accumulators.

    `STAGE_STOP` and `M` are both degenerate in vanilla F4 (`STAGE_STOP=L=2`,
    `M=1`). They exist so the same kernel handles two extra uses:
      - `STAGE_STOP=1`: stop after the s=0 stage, for the sanity_check.py
        stage-1 isolation test (no twiddles, no second matmul).
      - `M>1` with `STORE_T=True`: F7's fused FFT-m_0+T3, writing the
        transposed (rows_outer, 256, M) layout the next level expects.

    STORE_T=False (M=1): natural (B, 256) row-major output.
    STORE_T=True  (M>1): transposed (B//M, 256, M) output for F7 fusion.

    Each stage's four-`tl.dot` is one `_cdot` call; cast its fp32 output to
    fp16 before the next stage.

    Dtype contract:
        Loads:           fp16
        Reshape/permute: fp16 (free)
        tl.dot inputs:   fp16, out_dtype=tl.float32  (use _cdot)
        Twiddle mul:     fp32 * fp16 -> fp32
        Inter-stage:     .to(tl.float16) before next iter's reshape
        Store:           fp16
    Forgetting the inter-stage cast doubles register pressure and passes the
    L=2 tolerance, but fails as soon as F6 stacks more stages.

    Triton 3.6 gotcha -- tl.permute requires LITERAL tuples:
        tl.permute(x, (1, 0, 2))                  # works
        perm = (1, 0, 2); tl.permute(x, perm)     # fails
    Inline each stage's permute tuple at the call site; don't store the
    schedule in a loop variable.
    """
    pid = tl.program_id(0)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < B
    offs_n = tl.arange(0, 256)

    # Load (BLOCK_B, 256) fp16, reshape to (b, d_0, d_1)
    x_off = offs_b[:, None] * 256 + offs_n[None, :]
    x_mask = mask_b[:, None]
    t_re = tl.load(x_re_ptr + x_off, mask=x_mask, other=0.0)
    t_im = tl.load(x_im_ptr + x_off, mask=x_mask, other=0.0)
    t_re = tl.reshape(t_re, (BLOCK_B, 16, 16))
    t_im = tl.reshape(t_im, (BLOCK_B, 16, 16))

    # F_16 DFT matrix
    i0 = tl.arange(0, 16)
    F_off = i0[:, None] * 16 + i0[None, :]
    F_re = tl.load(F_re_ptr + F_off)
    F_im = tl.load(F_im_ptr + F_off)

    # Permute (b, d_0, d_1) -> (b, d_1, d_0) -- active digit at last axis (K of tl.dot)
    t_re = tl.permute(t_re, (0, 2, 1))
    t_im = tl.permute(t_im, (0, 2, 1))
    # No twiddle multiply at s = 0
    a_re = tl.reshape(t_re, (BLOCK_B * 16, 16))
    a_im = tl.reshape(t_im, (BLOCK_B * 16, 16))
    o_re, o_im = _cdot(a_re, a_im, F_re, F_im)
    t_re = tl.reshape(o_re.to(tl.float16), (BLOCK_B, 16, 16))   # (b, d_1, e_1)
    t_im = tl.reshape(o_im.to(tl.float16), (BLOCK_B, 16, 16))

    if STAGE_STOP > 1:
        # transform d_1 -> e_0
        # Permute (b, d_1, e_1) -> (b, e_1, d_1): active digit at last axis.
        t_re = tl.permute(t_re, (0, 2, 1))
        t_im = tl.permute(t_im, (0, 2, 1))

        # Stage-1 twiddle tw[1, m=d_1, c=e_1]. Tile axes are (b, e_1, d_1), so
        # load with rows=e_1, cols=d_1 -- transpose access to tw[1]
        tw_off = 256 + i0[None, :] * 16 + i0[:, None]
        tw_re_s = tl.load(tw_re_ptr + tw_off).to(tl.float32)
        tw_im_s = tl.load(tw_im_ptr + tw_off).to(tl.float32)

        # Complex mult in fp32, cast back to fp16 before next dot
        tr = t_re.to(tl.float32)
        ti = t_im.to(tl.float32)
        wr = tl.reshape(tw_re_s, (1, 16, 16))
        wi = tl.reshape(tw_im_s, (1, 16, 16))
        u_re = tr * wr - ti * wi
        u_im = tr * wi + ti * wr
        t_re = u_re.to(tl.float16)
        t_im = u_im.to(tl.float16)

        a_re = tl.reshape(t_re, (BLOCK_B * 16, 16))
        a_im = tl.reshape(t_im, (BLOCK_B * 16, 16))
        o_re, o_im = _cdot(a_re, a_im, F_re, F_im)
        t_re = tl.reshape(o_re.to(tl.float16), (BLOCK_B, 16, 16))   # (b, e_1, e_0)
        t_im = tl.reshape(o_im.to(tl.float16), (BLOCK_B, 16, 16))

    # Permute to natural output. For L=2 STAGE_STOP=2: (b, e_1, e_0) ->
    # (b, e_0, e_1) so linear = b*256 + e_0*16 + e_1 = b*256 + k. For
    # STAGE_STOP=1: (b, d_1, e_1) -> (b, e_1, d_1) matches torch.fft.fft on
    # dim=1 of the (B, 16, 16) input view
    t_re = tl.permute(t_re, (0, 2, 1))
    t_im = tl.permute(t_im, (0, 2, 1))

    t_re = tl.reshape(t_re, (BLOCK_B, 256))
    t_im = tl.reshape(t_im, (BLOCK_B, 256))

    # Natural (B, 256) or fused-T3 (B//M, 256, M) store layout
    if STORE_T:
        b_outer = offs_b // M
        m_idx = offs_b % M
        y_off = b_outer[:, None] * (256 * M) + offs_n[None, :] * M + m_idx[:, None]
    else:
        y_off = offs_b[:, None] * 256 + offs_n[None, :]

    y_mask = mask_b[:, None]
    tl.store(y_re_ptr + y_off, t_re, mask=y_mask)
    tl.store(y_im_ptr + y_off, t_im, mask=y_mask)


# =============================================================================
# dft_kernel: padded length-R DFT for the small chunks (R in {2, 4, 8, 16})
# =============================================================================

@triton.jit
def dft_kernel(
    x_re_ptr, x_im_ptr,     # (rows, R) fp16
    y_re_ptr, y_im_ptr,     # (rows, R) or (rows//M, R, M) fp16
    M_re_ptr, M_im_ptr,     # (16, 16) fp16 padded-R DFT matrix
    rows, M,
    R: tl.constexpr,
    BLOCK_B: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Padded length-R DFT via a (16, 16) tl.dot. STORE_T toggles natural
    vs transposed output (same pattern as f4_kernel_L2).

    One `_cdot(x_re, x_im, MT_re, MT_im)` call replaces the four `tl.dot`
    expansions; cast its fp32 result to fp16 on store.
    """
    pid = tl.program_id(0)
    offs_b = pid * BLOCK_B + tl.arange(0, BLOCK_B)
    mask_b = offs_b < rows

    offs_n = tl.arange(0, 16)
    mask_n = offs_n < R

    # Load (BLOCK_B, R) input padded to (BLOCK_B, 16) with zeros for n >= R
    x_off = offs_b[:, None] * R + offs_n[None, :]
    x_mask = mask_b[:, None] & mask_n[None, :]
    x_re = tl.load(x_re_ptr + x_off, mask=x_mask, other=0.0)
    x_im = tl.load(x_im_ptr + x_off, mask=x_mask, other=0.0)

    # Load M^T (16, 16):  MT[n, k] = M[k, n] = M_re_ptr[k*16 + n]
    MT_off = tl.arange(0, 16)[None, :] * 16 + tl.arange(0, 16)[:, None]
    MT_re = tl.load(M_re_ptr + MT_off)
    MT_im = tl.load(M_im_ptr + MT_off)

    # Y = X @ M^T as one complex matmul
    y_re_f32, y_im_f32 = _cdot(x_re, x_im, MT_re, MT_im)
    y_out_re = y_re_f32.to(tl.float16)
    y_out_im = y_im_f32.to(tl.float16)

    # Store: natural (rows, R) or fused-T (rows//M, R, M); keep only the first
    # R output columns (rows >= R are aliased copies the DFT pad produces)
    y_mask = mask_b[:, None] & mask_n[None, :]
    if STORE_T:
        b_outer = offs_b // M
        m_idx = offs_b % M
        y_off = b_outer[:, None] * (R * M) + offs_n[None, :] * M + m_idx[:, None]
    else:
        y_off = offs_b[:, None] * R + offs_n[None, :]

    tl.store(y_re_ptr + y_off, y_out_re, mask=y_mask)
    tl.store(y_im_ptr + y_off, y_out_im, mask=y_mask)


# =============================================================================
# bailey_scale_kernel: elementwise w_N^{n1 kM} multiply with optional fused T2
# =============================================================================

@triton.jit
def bailey_scale_kernel(
    x_re_ptr, x_im_ptr,     # (rows*m0*M,) fp16 input (logical (rows, m0, M))
    y_re_ptr, y_im_ptr,     # (rows*m0*M,) fp16 output ((rows, m0, M) or (rows, M, m0))
    tw_re_ptr, tw_im_ptr,   # (m0, M) fp16
    m0, M,
    BLOCK_M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Elementwise complex multiply by bt[n1, kM] over the (rows, m0, M) view.
    fp32 arithmetic, fp16 result. STORE_T=True fuses with a transpose to
    produce (rows, M, m0).

    Grid: (cdiv(m0, BLOCK_M0), cdiv(M, BLOCK_M), rows).
    """
    pid_n1 = tl.program_id(0)
    pid_kM = tl.program_id(1)
    pid_r  = tl.program_id(2)

    offs_n1 = pid_n1 * BLOCK_M0 + tl.arange(0, BLOCK_M0)
    offs_kM = pid_kM * BLOCK_M  + tl.arange(0, BLOCK_M)
    mask = (offs_n1[:, None] < m0) & (offs_kM[None, :] < M)

    # Twiddle bt[n1, kM] (shape (m0, M))
    # offset = n1 * M + kM
    tw_off = offs_n1[:, None] * M + offs_kM[None, :]
    tw_re = tl.load(tw_re_ptr + tw_off, mask=mask)
    tw_im = tl.load(tw_im_ptr + tw_off, mask=mask)

    # Input x[r, n1, kM] (shape (rows, m0, M))
    # offset = r*m0*M + n1*M + kM
    base = pid_r * m0 * M
    in_off = base + offs_n1[:, None] * M + offs_kM[None, :]
    x_re = tl.load(x_re_ptr + in_off, mask=mask)
    x_im = tl.load(x_im_ptr + in_off, mask=mask)

    # multiply in fp32
    xr = x_re.to(tl.float32)
    xi = x_im.to(tl.float32)
    tr = tw_re.to(tl.float32)
    ti = tw_im.to(tl.float32)
    y_re = (xr * tr - xi * ti).to(x_re.dtype)
    y_im = (xr * ti + xi * tr).to(x_re.dtype)

    # Output address: natural (rows, m0, M) or fused-T2 (rows, M, m0)
    if STORE_T:
        out_off = base + offs_kM[None, :] * m0 + offs_n1[:, None]
    else:
        out_off = in_off

    tl.store(y_re_ptr + out_off, y_re, mask=mask)
    tl.store(y_im_ptr + out_off, y_im, mask=mask)


# =============================================================================
# Thin launch wrappers -- GIVEN, do not edit
# =============================================================================

def _transpose(in_re, in_im, out_re, out_im, B, R, C):
    """Logical (B, R, C) -> (B, C, R) transpose, paired re/im."""
    grid = (triton.cdiv(R, TRANSPOSE_BLOCK), triton.cdiv(C, TRANSPOSE_BLOCK), B)
    transpose_kernel[grid](
        in_re, in_im, out_re, out_im, R, C,
        BLOCK_R=TRANSPOSE_BLOCK, BLOCK_C=TRANSPOSE_BLOCK,
    )


def _fft_chunk(in_re, in_im, out_re, out_im, rows, m, plan, M=1, store_t=False):
    """Length-m FFT over `rows` contiguous (rows, m) signals.

    M / store_t control the output layout:
      store_t=False, M=1: natural (rows, m) row-major (F6 leaf path)
      store_t=True,  M>1: transposed (rows//M, m, M) (F7 fused FFT-m0+T3)
    """
    if m == 256:
        f4_plan = plan['f4_plan']
        f4_kernel_L2[(triton.cdiv(rows, F4_L2_BLOCK_B),)](
            in_re.view(rows, 256), in_im.view(rows, 256),
            out_re.view(rows, 256), out_im.view(rows, 256),
            f4_plan['F_re'], f4_plan['F_im'],
            f4_plan['tw_re'], f4_plan['tw_im'],
            rows, M,
            BLOCK_B=F4_L2_BLOCK_B, STAGE_STOP=f4_plan['L'], STORE_T=store_t,
            num_warps=4, num_stages=1,
        )
    else:
        M_re, M_im = plan['dft_mats'][m]
        dft_kernel[(triton.cdiv(rows, DFT_BLOCK_B),)](
            in_re.view(rows, m), in_im.view(rows, m),
            out_re.view(rows, m), out_im.view(rows, m),
            M_re, M_im, rows, M,
            R=m, BLOCK_B=DFT_BLOCK_B, STORE_T=store_t,
        )


def _scale(in_re, in_im, out_re, out_im, rows, m0, M, twr, twi, store_t=False):
    """Bailey scale over logical (rows, m0, M)."""
    grid = (triton.cdiv(m0, SCALE_BLOCK), triton.cdiv(M, SCALE_BLOCK), rows)
    bailey_scale_kernel[grid](
        in_re, in_im, out_re, out_im, twr, twi,
        m0, M, BLOCK_M0=SCALE_BLOCK, BLOCK_M=SCALE_BLOCK, STORE_T=store_t,
    )


def _lookup_tw(plan, m0, M, N_i):
    """Find the precomputed Bailey twiddle table for (m0, M, N_i) in plan['tw']."""
    for (a, b, n, tr, ti) in plan['tw']:
        if a == m0 and b == M and n == N_i:
            return tr, ti
    raise KeyError(f"no twiddle table for (m0={m0}, M={M}, N={N_i})")


# =============================================================================
# F3 pipeline: 4-step Bailey six-step (T1 -> F2-A -> T2 -> F2-B)
# =============================================================================

def f3_launch(in_re, in_im, out_re, out_im, mid_re, mid_im, plan, B):
    """Run the 4-step F3 pipeline. Buffer ping-pong: in -> mid -> out -> mid
    -> out. Bailey scale + T2 fuse into one bailey_scale_kernel launch
    (STORE_T=True), and the would-be T3 is absorbed by F2-B (STRIDED_STORE=True).

    Steps:
      1. T1 (transpose): x[b, n2, n1] -> A[b, n1, n2]                in  -> mid
      2. F2-A:           length-N2 FFT over (B*N1) signals           mid -> out
      3. Scale + T2:     bt[n1, k2] multiply with fused transpose    out -> mid
      4. F2-B:           length-N1 FFT over (B*N2) signals,
                         strided store (post-T3 layout)              mid -> out
    """
    N1 = plan['N1']
    N2 = plan['N2']
    N = plan['N']
    LOG2_N1 = plan['LOG2_N1']
    LOG2_N2 = plan['LOG2_N2']

    # T1:  Logical (B, N2, N1) -> (B, N1, N2).
    _transpose(in_re, in_im, mid_re, mid_im, B, N2, N1)

    # F2-A:  Length-N2 FFT over (B*N1) signals.
    f2_kernel[(B * N1,)](
        mid_re, mid_im,
        out_re, out_im,
        plan['tw_re_n2'], plan['tw_im_n2'],
        plan['perm_n2'],
        plan['tw_re_n2'], plan['tw_im_n2'],   # bt_*_ptr sentinel
        1, 0,                                  # OUTER_DIM, N_TOTAL unused
        N=N2, LOG2_N=LOG2_N2,
        BAILEY_EPILOGUE=False, STRIDED_STORE=False,
        num_warps=4, num_stages=2,
    )

    # Scale + fused T2:  (B, N1, N2) -> (B, N2, N1) with bt[n1, k2] applied.
    _scale(out_re, out_im, mid_re, mid_im, B, N1, N2,
           plan['bt_re'], plan['bt_im'], store_t=True)

    # F2-B:  Length-N1 FFT over (B*N2) signals, strided store fuses T3.
    f2_kernel[(B * N2,)](
        mid_re, mid_im,
        out_re, out_im,
        plan['tw_re_n1'], plan['tw_im_n1'],
        plan['perm_n1'],
        plan['tw_re_n1'], plan['tw_im_n1'],   # bt_*_ptr sentinel
        N2, N,                                 # OUTER_DIM=N2, N_TOTAL=N
        N=N1, LOG2_N=LOG2_N1,
        BAILEY_EPILOGUE=False, STRIDED_STORE=True,
        num_warps=4, num_stages=2,
    )


# =============================================================================
# F5 pipeline: 6-step Bailey at N1=N2=256 with F4 as inner FFT
# =============================================================================

def f5_launch(in_re, in_im, b0_re, b0_im, b1_re, b1_im, b2_re, b2_im, plan, B):
    """Run the 6-step F5 pipeline at N = 65536 = 256 * 256.

    Buffer ping-pong: in -> b0 -> b1 -> b0 -> b1 -> b2 -> b0 (final).
    The Bailey twiddle is NOT fused into F4 (F4 stays unmodified), so this is
    6 launches; F7 generalizes the fusion idea recursively.

    Steps:
      1. T1:    x[b, n2, n1] -> A[b, n1, n2]                in -> b0
      2. FFT-A: length-256 FFT along last axis              b0 -> b1
      3. Scale: bt[n1, k2] multiply                         b1 -> b0
      4. T2:    Z[b, n1, k2] -> Z'[b, k2, n1]               b0 -> b1
      5. FFT-B: length-256 FFT along last axis              b1 -> b2
      6. T3:    V[b, k2, k1] -> X[b, k1, k2]                b2 -> b0
    """
    N1 = plan['N1']
    N2 = plan['N2']

    # T1: (B, N2, N1) -> (B, N1, N2).
    _transpose(in_re, in_im, b0_re, b0_im, B, N2, N1)

    # FFT-A: length-N2 FFT over (B*N1) signals.
    _fft_chunk(b0_re, b0_im, b1_re, b1_im, B * N1, N2, plan)

    # Scale: elementwise multiply by bt[n1, k2] over (B, N1, N2).
    _scale(b1_re, b1_im, b0_re, b0_im, B, N1, N2,
           plan['bt_re'], plan['bt_im'])

    # T2: (B, N1, N2) -> (B, N2, N1).
    _transpose(b0_re, b0_im, b1_re, b1_im, B, N1, N2)

    # FFT-B: length-N1 FFT over (B*N2) signals.
    _fft_chunk(b1_re, b1_im, b2_re, b2_im, B * N2, N1, plan)

    # T3: (B, N2, N1) -> (B, N1, N2).  Final lands in b0
    _transpose(b2_re, b2_im, b0_re, b0_im, B, N2, N1)


# =============================================================================
# F6 / F7 recursion
# =============================================================================
# Per level i with chunks = [m_0, m_1, ..., m_{p-1}], M = prod(chunks[1:]):
#   T1 :       (rows, M, m_0) -> (rows, m_0, M)
#   recurse:   length-M FFT over (rows*m_0, M)
#   Scale :    y *= w_{N_i}^{n_1 k_M}            (n_1 = the m_0 digit)
#   T2 :       (rows, m_0, M) -> (rows, M, m_0)
#   FFT-m_0 :  length-m_0 FFT over (rows*M, m_0)
#   T3 :       (rows, M, m_0) -> (rows, m_0, M)   [F6 only; F7 fuses]

def _f6_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Recursive 2-factor Bailey split. Leaf (len(chunks)==1) is one
    _fft_chunk call; non-leaf is the 6-step pipeline above.

    Returns the (re, im) cycler-managed buffers holding the (rows, prod(chunks))
    FFT result.
    """
    if len(chunks) == 1:
        out_re, out_im = cyc.next()
        _fft_chunk(cur_re, cur_im, out_re, out_im, rows, chunks[0], plan)
        return out_re, out_im

    m0 = chunks[0]
    inner = chunks[1:]
    M = math.prod(inner)
    Ni = m0 * M
    bt_re, bt_im = _lookup_tw(plan, m0, M, Ni)

    # T1: (rows, M, m0) -> (rows, m0, M).
    t1_re, t1_im = cyc.next()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)

    # Recurse length-M FFT over (rows*m0, M).
    rec_re, rec_im = _f6_rec(t1_re, t1_im, rows * m0, inner, plan, cyc)

    # Scale bt[n1, kM] over (rows, m0, M).
    sc_re, sc_im = cyc.next()
    _scale(rec_re, rec_im, sc_re, sc_im, rows, m0, M, bt_re, bt_im)

    # T2: (rows, m0, M) -> (rows, M, m0).
    t2_re, t2_im = cyc.next()
    _transpose(sc_re, sc_im, t2_re, t2_im, rows, m0, M)

    # FFT-m0: length-m0 FFT over (rows*M, m0).
    fft_re, fft_im = cyc.next()
    _fft_chunk(t2_re, t2_im, fft_re, fft_im, rows * M, m0, plan)

    # T3: (rows, M, m0) -> (rows, m0, M).
    t3_re, t3_im = cyc.next()
    _transpose(fft_re, fft_im, t3_re, t3_im, rows, M, m0)

    return t3_re, t3_im


def _f7_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Same recursion as _f6_rec but with Scale+T2 fused (store_t=True on
    bailey_scale_kernel) and FFT-m_0+T3 fused (store_t=True, M=M on the inner
    FFT kernel). Output should be bitwise-equal to _f6_rec.
    """
    if len(chunks) == 1:
        out_re, out_im = cyc.next()
        _fft_chunk(cur_re, cur_im, out_re, out_im, rows, chunks[0], plan)
        return out_re, out_im

    m0 = chunks[0]
    inner = chunks[1:]
    M = math.prod(inner)
    Ni = m0 * M
    bt_re, bt_im = _lookup_tw(plan, m0, M, Ni)

    # T1: (rows, M, m0) -> (rows, m0, M)
    t1_re, t1_im = cyc.next()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)

    # Recurse length-M FFT over (rows*m0, M)
    rec_re, rec_im = _f7_rec(t1_re, t1_im, rows * m0, inner, plan, cyc)

    # Scale + T2 fused: bt[n1, kM] over (rows, m0, M) writes (rows, M, m0)
    sc_re, sc_im = cyc.next()
    _scale(rec_re, rec_im, sc_re, sc_im, rows, m0, M, bt_re, bt_im, store_t=True)

    # FFT-m0 + T3 fused: length-m0 FFT over (rows*M, m0) writes (rows, m0, M)
    fft_re, fft_im = cyc.next()
    _fft_chunk(sc_re, sc_im, fft_re, fft_im, rows * M, m0, plan, M=M, store_t=True)

    return fft_re, fft_im
