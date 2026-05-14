"""Per-helper unit tests for twiddles.py.

Each helper is checked against hardcoded reference values at a small N. These
tests catch the common bugs (sign convention, dtype, shape, axis order)
before any kernel runs. If sanity_check.py FAILs and you're not sure whether
the kernel or the twiddle is wrong, run this first.

Usage:
    python twiddle_check.py            # check your twiddles.py
    FFT_REF=1 python twiddle_check.py  # confirm the references pass first
"""

import math
import os

import torch


USE_REF = os.environ.get('FFT_REF', '0') == '1'
if USE_REF:
    import twiddles_ref as twiddles
else:
    import twiddles


def _max_diff(actual_re, actual_im, exp_re, exp_im, device):
    er = torch.tensor(exp_re, dtype=torch.float32, device=device)
    ei = torch.tensor(exp_im, dtype=torch.float32, device=device)
    dr = (actual_re.to(torch.float32) - er).abs().max().item()
    di = (actual_im.to(torch.float32) - ei).abs().max().item()
    return max(dr, di)


# =============================================================================
# Per-helper checks. Each returns (verdict, detail) where verdict is one of
# 'PASS', 'FAIL', 'SKIP' and detail is a short string (None on PASS).
# =============================================================================

def check_radix2_twiddles(device):
    # N=8, fp32 default. w_N^k = exp(-2*pi*i * k / 8) for k = 0..3.
    SQ = 1.0 / math.sqrt(2.0)
    exp_re = [1.0,  SQ,  0.0, -SQ]
    exp_im = [0.0, -SQ, -1.0, -SQ]
    try:
        re, im = twiddles.make_radix2_twiddles(8, device=device)
    except NotImplementedError as e:
        return 'SKIP', f'not implemented: {e}'

    if tuple(re.shape) != (4,) or tuple(im.shape) != (4,):
        return 'FAIL', f'shape: want (4,), got re={tuple(re.shape)} im={tuple(im.shape)}'
    if re.dtype != torch.float32:
        return 'FAIL', f'dtype: want float32 (default), got {re.dtype}'
    d = _max_diff(re, im, exp_re, exp_im, device)
    if d < 1e-5:
        return 'PASS', None
    return 'FAIL', f'numerical mismatch at N=8: max diff = {d:.3e} (likely sign convention or off-by-one)'


def check_bit_reversal_perm(device):
    # N=8: 3-bit reversal. 0,1,2,3,4,5,6,7 -> 0,4,2,6,1,5,3,7.
    expected = [0, 4, 2, 6, 1, 5, 3, 7]
    try:
        perm = twiddles.bit_reversal_perm(8, device=device)
    except NotImplementedError as e:
        return 'SKIP', f'not implemented: {e}'

    if tuple(perm.shape) != (8,):
        return 'FAIL', f'shape: want (8,), got {tuple(perm.shape)}'
    if perm.dtype != torch.int32:
        return 'FAIL', f'dtype: want int32, got {perm.dtype}'
    actual = perm.cpu().tolist()
    if actual == expected:
        return 'PASS', None
    return 'FAIL', f'want {expected}, got {actual}'


def check_dft_matrix(device):
    # N=4: W[j, k] = exp(-2*pi*i * j*k / 4), fp16 default.
    exp_re = [
        [1.0,  1.0,  1.0,  1.0],
        [1.0,  0.0, -1.0,  0.0],
        [1.0, -1.0,  1.0, -1.0],
        [1.0,  0.0, -1.0,  0.0],
    ]
    exp_im = [
        [0.0,  0.0,  0.0,  0.0],
        [0.0, -1.0,  0.0,  1.0],
        [0.0,  0.0,  0.0,  0.0],
        [0.0,  1.0,  0.0, -1.0],
    ]
    try:
        re, im = twiddles.make_dft_matrix(4, device=device)
    except NotImplementedError as e:
        return 'SKIP', f'not implemented: {e}'

    if tuple(re.shape) != (4, 4):
        return 'FAIL', f'shape: want (4, 4), got {tuple(re.shape)}'
    if re.dtype != torch.float16:
        return 'FAIL', f'dtype: want float16 (default), got {re.dtype}'
    d = _max_diff(re, im, exp_re, exp_im, device)
    if d < 2e-3:
        return 'PASS', None
    return 'FAIL', f'numerical mismatch at N=4: max diff = {d:.3e} (likely sign convention or transposed)'


def check_bailey_cross_twiddles(device):
    # m0=2, M=2, N=4: bt[n1, kM] = exp(-2*pi*i * n1*kM / 4), fp16 default.
    # bt = [[1, 1], [1, -i]]
    exp_re = [[1.0, 1.0], [1.0, 0.0]]
    exp_im = [[0.0, 0.0], [0.0, -1.0]]
    try:
        re, im = twiddles.make_bailey_cross_twiddles(2, 2, 4, device=device)
    except NotImplementedError as e:
        return 'SKIP', f'not implemented: {e}'

    if tuple(re.shape) != (2, 2):
        return 'FAIL', f'shape: want (2, 2), got {tuple(re.shape)}'
    if re.dtype != torch.float16:
        return 'FAIL', f'dtype: want float16 (default), got {re.dtype}'
    d = _max_diff(re, im, exp_re, exp_im, device)
    if d < 2e-3:
        return 'PASS', None
    return 'FAIL', f'numerical mismatch at (m0=2, M=2, N=4): max diff = {d:.3e}'


def check_dft_R_padded(device):
    # R=4: full matrix is (16, 16) fp16; M[:4, :4] must equal F_4.
    # Columns 4..15 may be anything (kernel zero-pads input there), so only
    # the (R, R) corner is checked.
    exp_re_corner = [
        [1.0,  1.0,  1.0,  1.0],
        [1.0,  0.0, -1.0,  0.0],
        [1.0, -1.0,  1.0, -1.0],
        [1.0,  0.0, -1.0,  0.0],
    ]
    exp_im_corner = [
        [0.0,  0.0,  0.0,  0.0],
        [0.0, -1.0,  0.0,  1.0],
        [0.0,  0.0,  0.0,  0.0],
        [0.0,  1.0,  0.0, -1.0],
    ]
    try:
        re, im = twiddles.make_dft_R_padded(4, device=device)
    except NotImplementedError as e:
        return 'SKIP', f'not implemented: {e}'

    if tuple(re.shape) != (16, 16):
        return 'FAIL', f'shape: want (16, 16), got {tuple(re.shape)}'
    if re.dtype != torch.float16:
        return 'FAIL', f'dtype: want float16, got {re.dtype}'
    d = _max_diff(re[:4, :4], im[:4, :4], exp_re_corner, exp_im_corner, device)
    if d < 2e-3:
        return 'PASS', None
    return 'FAIL', f'F_4 corner mismatch at make_dft_R_padded(4): max diff = {d:.3e}'


def check_radix16_twiddles(device):
    # N=256, L=2:
    #   tw[0, :, :] should be all ones (kernel skips multiply on s == 0).
    #   tw[1, m, c] = exp(-2*pi*i * m * c / 256) at L=2 (single-digit column).
    try:
        re, im = twiddles.make_radix16_twiddles(256, device=device)
    except NotImplementedError as e:
        return 'SKIP', f'not implemented: {e}'

    if tuple(re.shape) != (2, 16, 16):
        return 'FAIL', f'shape: want (2, 16, 16), got {tuple(re.shape)}'
    if re.dtype != torch.float16:
        return 'FAIL', f'dtype: want float16, got {re.dtype}'

    # Stage 0: all ones.
    re0 = re[0].to(torch.float32).cpu()
    im0 = im[0].to(torch.float32).cpu()
    d0 = max((re0 - 1.0).abs().max().item(), im0.abs().max().item())
    if d0 > 1e-3:
        return 'FAIL', f'stage-0 slice is not all ones (kernel skips multiply at s == 0); max diff = {d0:.3e}'

    # Stage 1: spot-check entries.
    spots = [
        (0, 0,  1.0,                              0.0),
        (1, 0,  1.0,                              0.0),
        (0, 5,  1.0,                              0.0),
        (1, 1,  math.cos(-2 * math.pi *  1 / 256), math.sin(-2 * math.pi *  1 / 256)),
        (2, 3,  math.cos(-2 * math.pi *  6 / 256), math.sin(-2 * math.pi *  6 / 256)),
        (4, 4,  math.cos(-2 * math.pi * 16 / 256), math.sin(-2 * math.pi * 16 / 256)),
    ]
    re1 = re[1].to(torch.float32).cpu()
    im1 = im[1].to(torch.float32).cpu()
    for m, c, er, ei in spots:
        ar = re1[m, c].item()
        ai = im1[m, c].item()
        if abs(ar - er) > 5e-3 or abs(ai - ei) > 5e-3:
            return 'FAIL', (
                f'stage-1 entry [{m},{c}]: want ({er:+.4f}, {ei:+.4f}j), '
                f'got ({ar:+.4f}, {ai:+.4f}j) -- check sign convention and the '
                f'column-axis labeling at s=1'
            )
    return 'PASS', None


# =============================================================================

def main():
    if torch.cuda.is_available():
        device = 'cuda'
        dev_name = torch.cuda.get_device_name(0)
    else:
        device = 'cpu'
        dev_name = 'cpu (no CUDA)'

    print(f'device:  {dev_name}')
    print(f'torch:   {torch.__version__}')
    print(f'FFT_REF: {"1 (reference modules)" if USE_REF else "0 (student modules)"}')
    print()

    checks = [
        ('make_radix2_twiddles',       check_radix2_twiddles),
        ('bit_reversal_perm',          check_bit_reversal_perm),
        ('make_dft_matrix',            check_dft_matrix),
        ('make_bailey_cross_twiddles', check_bailey_cross_twiddles),
        ('make_dft_R_padded',          check_dft_R_padded),
        ('make_radix16_twiddles',      check_radix16_twiddles),
    ]

    print(f"  {'helper':<30}  {'verdict':<6}  detail")
    print(f"  {'-'*30}  {'-'*6}  {'-'*60}")
    for name, fn in checks:
        try:
            verdict, detail = fn(device)
        except Exception as e:
            verdict, detail = 'FAIL', f'{type(e).__name__}: {e}'
        print(f"  {name:<30}  {verdict:<6}  {detail or ''}")


if __name__ == '__main__':
    main()
