# filename: even_odd_zeta_fft.py
from __future__ import annotations
from mpmath import mp
import numbers
import csv
from datetime import datetime
import numpy as np
import time
from typing import Optional, Tuple

# Optional DCT support (falls back to FFT if SciPy not available)
try:
    try:
        from scipy.fft import dct as _dct, idct as _idct  # type: ignore
    except Exception:  # pragma: no cover - compatibility path
        from scipy.fftpack import dct as _dct, idct as _idct  # type: ignore
    _HAS_DCT = True
except Exception:  # pragma: no cover - SciPy missing
    _HAS_DCT = False

# High precision for digit decisions
mp.dps = 80

def exact_even_sum() -> numbers.Real:
    return (mp.pi ** 2) / 24

def exact_odd_sum() -> numbers.Real:
    return (mp.pi ** 2) / 8

def get_digit(x: numbers.Real, d: int) -> int:
    assert d >= 1
    s = mp.nstr(x, d + 1)
    if '.' not in s:
        return 0 if d > 0 else int(s)
    frac = s.split('.')[1]
    return int(frac[d - 1]) if d <= len(frac) else 0

def decision_digit_even(d: int, t: int) -> bool:
    # Rigorous and fast: closed form
    mp.dps = d + 10
    return get_digit(exact_even_sum(), d) == t

def decision_digit_odd(d: int, t: int) -> bool:
    # Rigorous and fast: closed form
    mp.dps = d + 10
    return get_digit(exact_odd_sum(), d) == t

def partial_even(n: int) -> numbers.Real:
    # sum_{k=1}^n 1/(2k)^2 = (1/4)*sum_{k=1}^n 1/k^2
    s = mp.mpf('0')
    for k in range(1, n + 1):
        s += 1 / (k * k)
    return s / 4

def partial_odd(n: int) -> numbers.Real:
    # sum_{k=1}^n 1/(2k-1)^2
    s = mp.mpf('0')
    for k in range(1, n + 1):
        denom = 2 * k - 1
        s += 1 / (denom * denom)
    return s

def time_digit_decisions(d_list=(8, 16), t=5) -> None:
    print("Digit-decision (both are in P; using closed forms):")
    for d in d_list:
        t0 = time.time()
        r_even = decision_digit_even(d, t)
        t1 = time.time()
        r_odd = decision_digit_odd(d, t)
        t2 = time.time()
        print(f"d={d}: even=={t}? {r_even}  ({t1 - t0:.6f}s), odd=={t}? {r_odd}  ({t2 - t1:.6f}s)")

def fft2_convolution_sum(a: np.ndarray, b: np.ndarray) -> float:
    # 2D circular convolution sum via FFT2: sum of (a ⊗ b) equals (sum a)*(sum b)
    # Shown via FFT as a structure/symmetry demo.
    A = np.fft.fft2(a)
    B = np.fft.fft2(b)
    C = A * B
    conv = np.fft.ifft2(C).real
    return float(conv.sum())

def symmetry_demo_fft(N: int = 512) -> None:
    # Build structured kernels. Even-indexed weights are more “regular” (period-2 mask).
    k = np.arange(1, N + 1, dtype=np.float64)
    even_w = (1.0 / (2.0 * k) ** 2)  # values at even indices; zero-pad to N if desired
    odd_w = (1.0 / (2.0 * k - 1.0) ** 2)

    # Build and process 2D grids sequentially to reduce peak memory
    n_fft = 1 << (N - 1).bit_length()  # next power of two
    pe = ((0, n_fft - N), (0, n_fft - N))

    print(f"FFT symmetry demo (N={N}->n_fft={n_fft}):")
    # Even
    t0 = time.time()
    grid_even = np.outer(even_w, even_w)
    even_pad = np.pad(grid_even, pe)
    del grid_even
    se = fft2_convolution_sum(even_pad, even_pad)
    t1 = time.time()
    print(f"even grid conv-sum via FFT2: {se:.12e}  time={t1 - t0:.4f}s")
    # Odd
    t2 = time.time()
    grid_odd = np.outer(odd_w, odd_w)
    odd_pad  = np.pad(grid_odd,  pe)
    del grid_odd
    so = fft2_convolution_sum(odd_pad, odd_pad)
    t3 = time.time()
    print(f"odd  grid conv-sum via FFT2: {so:.12e}  time={t3 - t2:.4f}s")
    # Note: mathematically sum(grid) = (sum(weights))^2. FFT here just illustrates structure-friendly scaling.

# --------------------
# FFT/DCT utilities and preprocessing controls
# --------------------

def _apply_window(signal: np.ndarray, window: Optional[str]) -> np.ndarray:
    if window is None:
        return signal
    w = window.lower()
    if w in ("hann", "hanning"):
        return signal * np.hanning(len(signal))
    # Unknown window: no-op
    return signal

def preprocess_sequence(sequence: np.ndarray, unit_norm: bool = True, remove_dc: bool = False, window: Optional[str] = None) -> np.ndarray:
    seq = np.asarray(sequence, dtype=np.float64)
    if remove_dc:
        seq = seq - float(np.mean(seq))
    seq = _apply_window(seq, window)
    if unit_norm:
        norm = float(np.linalg.norm(seq))
        if norm > 0.0:
            seq = seq / norm
    return seq

def _transform(signal: np.ndarray, basis: str) -> Tuple[np.ndarray, int]:
    basis_lc = basis.lower()
    if basis_lc == "rfft":
        return np.fft.rfft(signal), len(signal)
    if basis_lc == "dct":
        if not _HAS_DCT:
            print("[info] DCT basis requested but SciPy not available; falling back to rFFT.")
            return np.fft.rfft(signal), len(signal)
        return _dct(signal, type=2, norm="ortho"), len(signal)
    # Default
    return np.fft.rfft(signal), len(signal)

def _inverse_from_top_m(transform_coeffs: np.ndarray, keep_indices: np.ndarray, basis: str, signal_length: int) -> np.ndarray:
    basis_lc = basis.lower()
    if basis_lc == "rfft" or (basis_lc == "dct" and not _HAS_DCT):
        F_approx = np.zeros_like(transform_coeffs)
        F_approx[keep_indices] = transform_coeffs[keep_indices]
        return np.fft.irfft(F_approx, n=signal_length)
    if basis_lc == "dct":
        coeffs = np.zeros_like(transform_coeffs)
        coeffs[keep_indices] = transform_coeffs[keep_indices]
        return _idct(coeffs, type=2, norm="ortho")
    # Fallback
    F_approx = np.zeros_like(transform_coeffs)
    F_approx[keep_indices] = transform_coeffs[keep_indices]
    return np.fft.irfft(F_approx, n=signal_length)

def top_m_rel_error_generic(seq: np.ndarray, M: int, *, basis: str = "rfft", unit_norm: bool = True, remove_dc: bool = False, window: Optional[str] = None) -> float:
    prepared = preprocess_sequence(seq, unit_norm=unit_norm, remove_dc=remove_dc, window=window)
    T, N = _transform(prepared, basis)
    order = np.argsort(np.abs(T))[::-1]
    keep = order[: max(1, int(M))]
    recon = _inverse_from_top_m(T, keep, basis=basis, signal_length=N)
    numerator = float(np.linalg.norm(prepared - recon))
    denominator = float(np.linalg.norm(prepared))
    return (numerator / denominator) if denominator != 0.0 else 0.0

def symmetry_demo_fft_many(N_list=(256, 512, 1024, 2048, 4096)) -> None:
    for N in N_list:
        try:
            symmetry_demo_fft(N=N)
        except MemoryError:
            print(f"Skipping N={N} due to MemoryError")

def symmetry_demo_fft_many_extended() -> None:
    # Try larger sizes; rely on sequential processing and catch OOM
    sizes = (512, 1024, 2048, 4096, 8192)
    for N in sizes:
        try:
            symmetry_demo_fft(N=N)
        except MemoryError:
            print(f"Skipping N={N} due to MemoryError")

def convergence_compare(n_list=(10, 100, 1000)) -> None:
    # Empirical tails vs exact; highlights slightly larger odd tails under matched term counts.
    ee = exact_even_sum()
    eo = exact_odd_sum()
    print("Convergence comparison (partial vs exact):")
    for n in n_list:
        mp.dps = 40
        se = partial_even(n)
        so = partial_odd(n)
        err_e = abs(ee - se)
        err_o = abs(eo - so)
        # Cast to float for formatting in scientific notation
        err_e_f = float(err_e)
        err_o_f = float(err_o)
        ratio = err_o_f / err_e_f if err_e_f != 0.0 else float('inf')
        print(f"n={n:5d}  even_err≈{err_e_f:.6e}  odd_err≈{err_o_f:.6e}  ratio≈{ratio:.2f}")

def fft_compressibility_demo(
    L: int = 2048,
    M_list=(4, 8, 16, 32, 64),
    *,
    basis: str = "rfft",
    unit_norm: bool = True,
    remove_dc: bool = False,
    window: Optional[str] = None,
) -> None:
    """Quantify structure under FFT: how many largest Fourier modes are needed
    to approximate the even vs odd sequences to low error.

    seq_base[j] = 1/(j+1)^2, even/odd via period-2 masks. Lower relative error
    for a fixed M indicates more FFT-friendly structure (your framing).
    """
    j = np.arange(L, dtype=np.float64)
    base = 1.0 / ((j + 1.0) ** 2)
    even_mask = ((j + 1) % 2 == 0).astype(np.float64)
    odd_mask = 1.0 - even_mask
    seq_even = base * even_mask
    seq_odd = base * odd_mask

    print("FFT compressibility (lower is better):")
    for M in M_list:
        err_even = top_m_rel_error_generic(
            seq_even, M, basis=basis, unit_norm=unit_norm, remove_dc=remove_dc, window=window
        )
        err_odd = top_m_rel_error_generic(
            seq_odd, M, basis=basis, unit_norm=unit_norm, remove_dc=remove_dc, window=window
        )
        print(f"M={M:3d}  even_rel_err={err_even:.3e}  odd_rel_err={err_odd:.3e}  ratio={err_odd/err_even if err_even>0 else float('inf'):.2f}")

def _top_m_rel_error_rfft(seq: np.ndarray, M: int, *, unit_norm: bool = True, remove_dc: bool = False, window: Optional[str] = None) -> float:
    return top_m_rel_error_generic(seq, M, basis="rfft", unit_norm=unit_norm, remove_dc=remove_dc, window=window)

def fft_compressibility_proportional(
    L_list=(4096, 16384, 65536, 262144, 524288),
    fractions=(0.001, 0.002, 0.005, 0.01, 0.02),
    *,
    basis: str = "rfft",
    unit_norm: bool = True,
    remove_dc: bool = False,
    window: Optional[str] = None,
) -> None:
    """Serious test: keep M as a fraction of available rFFT bins.
    Ensures fair comparison across lengths; validates monotone improvement.
    Also writes CSV results to results_compressibility_proportional_*.csv in the project root
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results_compressibility_proportional_{timestamp}.csv"
    with open(out_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["L", "fraction", "M", "even_rel_err", "odd_rel_err", "ratio_odd_over_even"])
        print("\nProportional FFT compressibility (results saved to:", out_path, ")")
        for L in L_list:
            j = np.arange(L, dtype=np.float64)
            base = 1.0 / ((j + 1.0) ** 2)
            even_mask = ((j + 1) % 2 == 0).astype(np.float64)
            odd_mask = 1.0 - even_mask
            seq_even = base * even_mask
            seq_odd = base * odd_mask
            n_bins = L // 2 + 1
            last_even_err = None
            last_odd_err = None
            print(f"L={L}:")
            for frac in fractions:
                M = max(1, int(frac * n_bins))
                err_even = top_m_rel_error_generic(
                    seq_even, M, basis=basis, unit_norm=unit_norm, remove_dc=remove_dc, window=window
                )
                err_odd = top_m_rel_error_generic(
                    seq_odd, M, basis=basis, unit_norm=unit_norm, remove_dc=remove_dc, window=window
                )
                ratio = err_odd / err_even if err_even > 0 else float('inf')
                writer.writerow([L, frac, M, f"{err_even:.6e}", f"{err_odd:.6e}", f"{ratio:.3f}"])
                print(f"  frac={frac:.3f}  M={M:6d}  even={err_even:.3e}  odd={err_odd:.3e}  ratio={ratio:.2f}")
                # Sanity: error should not increase as M grows
                if last_even_err is not None:
                    assert err_even <= last_even_err + 1e-12, "even error not nonincreasing"
                    assert err_odd <= last_odd_err + 1e-12, "odd error not nonincreasing"
                last_even_err = err_even
                last_odd_err = err_odd

def fft_compressibility_sweep() -> None:
    # Extended lengths up to ~5e5 to emphasize divergence under fixed M
    for L in (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288):
        # Use a richer set of M values up to ~1024
        M_list = (8, 16, 32, 64, 128, 256, 512, 1024)
        print(f"\nCompressibility sweep for L={L}:")
        fft_compressibility_demo(L=L, M_list=M_list, basis="rfft", unit_norm=True, remove_dc=False, window=None)

def spectral_entropy(bits: int, seq: np.ndarray) -> float:
    # Shannon entropy of normalized power spectrum (base 2)
    # Use rFFT by default and honor preprocessing normalizations
    prepared = preprocess_sequence(seq, unit_norm=True, remove_dc=False, window=None)
    F = np.fft.rfft(prepared)
    power = np.abs(F) ** 2
    total = power.sum()
    if total == 0.0:
        return 0.0
    p = power / total
    # Avoid log(0)
    p = np.where(p > 0, p, 1.0)
    H = float(-(p * (np.log2(p))).sum())
    return H

def spectral_metrics(
    L_list=(1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144),
    *,
    remove_dc: bool = False,
) -> None:
    print("\nSpectral entropy and effective modes (2^H):")
    for L in L_list:
        j = np.arange(L, dtype=np.float64)
        base = 1.0 / ((j + 1.0) ** 2)
        even_mask = ((j + 1) % 2 == 0).astype(np.float64)
        odd_mask = 1.0 - even_mask
        seq_even = preprocess_sequence(base * even_mask, unit_norm=True, remove_dc=remove_dc, window=None)
        seq_odd = preprocess_sequence(base * odd_mask, unit_norm=True, remove_dc=remove_dc, window=None)
        H_even = spectral_entropy(2, seq_even)
        H_odd = spectral_entropy(2, seq_odd)
        eff_even = 2.0 ** H_even
        eff_odd = 2.0 ** H_odd
        print(f"L={L:4d}  H_even={H_even:.2f}  H_odd={H_odd:.2f}  eff_even≈{eff_even:.0f}  eff_odd≈{eff_odd:.0f}  ratio≈{(eff_odd/eff_even) if eff_even>0 else float('inf'):.2f}")

def min_modes_for_error(seq: np.ndarray, eps: float) -> int:
    # Use spectral energy threshold via Parseval (relative L2 error <= eps)
    prepared = preprocess_sequence(seq, unit_norm=True, remove_dc=False, window=None)
    F = np.fft.rfft(prepared)
    energy = np.abs(F) ** 2
    order = np.argsort(energy)[::-1]
    sorted_energy = energy[order]
    cum = np.cumsum(sorted_energy)
    total = float(energy.sum())
    if total == 0.0:
        return 0
    target = (1.0 - eps * eps) * total
    idx = np.searchsorted(cum, target, side='left')
    M = int(min(max(idx + 1, 0), len(sorted_energy)))
    return M

def min_modes_sweep(L_list=(1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072), eps_list=(1e-1, 3e-2, 1e-2)) -> None:
    print("\nMin modes M for target relative error eps (energy criterion):")
    for L in L_list:
        j = np.arange(L, dtype=np.float64)
        base = 1.0 / ((j + 1.0) ** 2)
        even_mask = ((j + 1) % 2 == 0).astype(np.float64)
        odd_mask = 1.0 - even_mask
        seq_even = preprocess_sequence(base * even_mask, unit_norm=True, remove_dc=False, window=None)
        seq_odd = preprocess_sequence(base * odd_mask, unit_norm=True, remove_dc=False, window=None)
        line = [f"L={L:4d}"]
        for eps in eps_list:
            M_even = min_modes_for_error(seq_even, eps)
            M_odd = min_modes_for_error(seq_odd, eps)
            ratio = (M_odd / M_even) if M_even > 0 else float('inf')
            line.append(f"eps={eps:.0e}: M_even={M_even:4d} M_odd={M_odd:4d} ratio={ratio:.2f}")
        print("  ".join(line))

def mod_q_compressibility(q_list=(3, 4, 6, 8), L: int = 65536, M_list=(16, 32, 64, 128, 256, 512, 1024)) -> None:
    print(f"\nMod-q residue-class compressibility at L={L}:")
    j = np.arange(L, dtype=np.float64)
    base = 1.0 / ((j + 1.0) ** 2)
    for q in q_list:
        print(f"q={q}:")
        for r in range(q):
            mask = (((j + 1) % q) == r).astype(np.float64)
            seq = preprocess_sequence(base * mask, unit_norm=True, remove_dc=False, window=None)
            # Entropy
            H = spectral_entropy(2, seq)
            eff = 2.0 ** H
            # One-line compressibility at a few M
            errs = []
            for M in M_list:
                F = np.fft.rfft(seq)
                idx = np.argsort(np.abs(F))[::-1][:M]
                F_keep = np.zeros_like(F)
                F_keep[idx] = F[idx]
                recon = np.fft.irfft(F_keep, n=L)
                num = np.linalg.norm(seq - recon)
                den = np.linalg.norm(seq)
                rel = float(num / den) if den != 0.0 else 0.0
                errs.append(rel)
            errs_str = " ".join(f"M={M:4d}:err={e:.2e}" for M, e in zip(M_list, errs))
            print(f"  r={r}: H={H:.2f} eff≈{eff:.0f}  {errs_str}")


def fixed_M_vs_length_divergence(
    M_list=(16, 32, 64, 128),
    L_powers=(10, 12, 14, 16, 18, 19),
    *,
    remove_dc: bool = False,
    window: Optional[str] = None,
    basis: str = "rfft",
) -> None:
    print("\nFixed-M vs length (divergence check; lower is better, ratio > 1 favors even):")
    for M in M_list:
        print(f"M={M}:")
        for k in L_powers:
            L = 1 << k
            try:
                j = np.arange(L, dtype=np.float64)
                base = 1.0 / ((j + 1.0) ** 2)
                even_mask = ((j + 1) % 2 == 0).astype(np.float64)
                odd_mask = 1.0 - even_mask
                seq_even = base * even_mask
                seq_odd = base * odd_mask
                err_even = top_m_rel_error_generic(
                    seq_even, M, basis=basis, unit_norm=True, remove_dc=remove_dc, window=window
                )
                err_odd = top_m_rel_error_generic(
                    seq_odd, M, basis=basis, unit_norm=True, remove_dc=remove_dc, window=window
                )
                ratio = (err_odd / err_even) if err_even > 0 else float('inf')
                print(f"  L=2^{k:2d} ({L:7d})  even={err_even:.3e}  odd={err_odd:.3e}  ratio={ratio:.2f}")
            except MemoryError:
                print(f"  L=2^{k} skipped due to MemoryError")

def main():
    # 1) Rigorous digit decisions (both P)
    time_digit_decisions(d_list=(8, 16), t=5)

    # 2) Convergence heuristic (analogy: odd tails slightly larger under same n)
    convergence_compare(n_list=(10, 100, 1000))

    # 3) FFT symmetry demo (analogy: structure aligns with power-of-2 efficiency)
    symmetry_demo_fft(N=256)

    # 4) FFT compressibility: even sequence is more compressible under FFT
    fft_compressibility_demo(L=2048, M_list=(4, 8, 16, 32, 64))

    # 5) Larger-scale runs up to 4096
    print("\n--- Larger FFT symmetry runs up to N=4096 ---")
    symmetry_demo_fft_many(N_list=(512, 1024, 2048, 4096))

    # 5b) Attempt extended 2D sizes (up to 8192)
    print("\n--- Extended FFT symmetry runs (attempt up to N=8192) ---")
    symmetry_demo_fft_many_extended()

    print("\n--- Compressibility sweeps up to L=4096 ---")
    fft_compressibility_sweep()

    # 6) Spectral metrics: entropy and effective modes
    spectral_metrics(L_list=(1024, 2048, 4096))

    # 7) Minimum modes to reach target error eps
    min_modes_sweep(L_list=(1024, 2048, 4096), eps_list=(1e-1, 3e-2, 1e-2))

    # 8) Mod-q residue class compressibility (zeta via Dirichlet characters)
    mod_q_compressibility(q_list=(3, 4, 6, 8), L=4096, M_list=(16, 32, 64, 128, 256))

    # 9) Proportional compressibility test with CSV logging
    fft_compressibility_proportional(L_list=(4096, 16384, 65536, 262144, 524288),
                                     fractions=(0.001, 0.002, 0.005, 0.01, 0.02))

    # 10) Long-horizon divergence at fixed M (optional, modest sizes by default)
    fixed_M_vs_length_divergence(M_list=(16, 32, 64, 128), L_powers=(10, 12, 14, 16))

if __name__ == "__main__":
    main()


