## Odd_VS_Even_Zeta_Substructure

[extended from original tests]

Even vs Odd Zeta Subsequences: Arithmetic Symmetry and Spectral Compressibility
Scope

Neutral summary of the parity split of (\sum 1/n^2) and the FFT-based measurements implemented here.
Introduction

We experiment with compressable sequences, link arithmetic symmetry (e.g., the p=2 Euler factor for now, more tests planned and some already implemented check back soon) to algorithmic compressibility under Fourier-type bases. Using the even/odd subsequences of the Basel series 1/n^2 as a canonical case, we show that the even subsequence is consistently more FFT‑compressible than the odd subsequence under matched budgets, we show that the bins needed for odd are ~7% more than even (read notes about the 7% claim).

### Overview Big Fluff
This repository studies arithmetic symmetry in zeta Dirichlet subsequences and its effect on spectral compressibility under Fourier-type transforms. Empirically, the odd subsequence of ζ(2) incurs a ~7% compressibility penalty versus the even subsequence at matched budgets. We generalize to ζ(s) for real s>1 and complex s, residue classes mod q, and provide CPU/GPU (optional torch) implementations.

### Motivation & Goals
- We explore how simple arithmetic structure (parity and residue-class masks tied to Euler factors/Dirichlet characters) changes spectral concentration and algorithmic compressibility under Fourier-type bases.
- Core hypothesis: small-period arithmetic masks act like sparse spectral combs, concentrating energy and lowering the top-M error; removing the p=2 factor (odd-only) diffuses this structure slightly, worsening compressibility.
- This is a structure/compressibility story inside one numeric family (zeta subsequences) – not a complexity-class claim.

### Key Questions We Answer
- Does parity (q=2) or general residue-class masking (mod q) measurably change compressibility of n^{-s} sequences?
- How stable is the odd vs even gap across lengths L, bases (DFT/DCT), windows, and preprocessing (DC removal)?
- What is the right way to quantify it: fixed-M error, fixed-ε minimal modes, entropy/effective modes, and proportional M?
- How does the effect extend beyond s=2: to ζ(s) with real s>1 and to the critical line s=1/2+it (using real parts for spectra)?
- What are the practical implications for FFT/DCT budgets in ML/DSP pipelines?

### Installation (uv + polars)
- Install uv (Windows PowerShell):
  - `iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex`
- Sync deps and create a virtualenv:
  - `uv sync`
- Run tests:
  - `uv run -m pytest -q`

### Quick Start
- CLI proportional experiment (ζ(2)) and plot:
  - `uv run python scripts/run_cli.py --lengths 4096,16384 --fractions 0.002,0.005 --plot`
  - Results saved to `results_compressibility_proportional_*.csv` and `plots/err_curves.png`.
- Residue-class entropy:
  - `uv run python scripts/residue_entropy.py --L 4096 --q-list 3,4,6,8 --plot`
  - Saves `residue_entropy_L4096.csv` and `plots/residue_entropy_L4096.png`.
- Critical line sweep (s = 1/2 + i t):
  - `uv run python scripts/critical_line_sweep.py --lengths 4096,8192 --fractions 0.002,0.005 --t-min 0 --t-max 50 --t-steps 26 --plot`
  - Saves `critical_line_ratio_*.csv` and `plots/ratio_vs_t.png`.
- Bootstrap CI for odd/even ratio:
  - `uv run python scripts/bootstrap_ci.py --L 4096,16384,65536 --fractions 0.002,0.005 --B 500`
  - Saves `bootstrap_ratio_ci.csv` with mean and confidence interval.
- Torch GPU example (optional):
  - `uv run python scripts/ml_example.py --L 16384 --M 128 --device cuda`

### Math Background
- ζ(s) = ∑_{n≥1} n^{-s}. For s=2 (Basel): ζ(2)=π²/6.
- Even split: ∑_{k≥1} (2k)^{-2} = (1/4)ζ(2) = π²/24.
- Odd split: ∑_{k≥1} (2k-1)^{-2} = (1-2^{-2})ζ(2) = π²/8.
- Euler product: ζ(s)=∏_{p}(1-p^{-s})^{-1}. Removing even indices corresponds to the factor (1-2^{-s}). This arithmetic mask imprints a period-2 structure that aligns better with Fourier bases, yielding improved compressibility.

### What We Measure
- Relative L2 reconstruction error after keeping top M Fourier/DCT modes.
- Proportional regime: M = fraction × (L/2+1).
- Spectral entropy H (base 2) and effective modes 2^H.
- Residue-class analyses mod q.

### Methodology (fixed-M vs fixed-ε)
- Fixed-M compressibility: keep the top-M coefficients by magnitude (optimal for L2 under orthonormal bases). Report relative L2 error
  \(\;\|x - x_M\|_2 / \|x\|_2\;\).
- Fixed-ε minimal modes: the smallest M such that
  \(\;\|x - x_M\|_2 / \|x\|_2 \le \varepsilon\;\).
  By Parseval, this is equivalent to an energy coverage threshold
  \(\;\n\sum_{k \in \text{kept}} |X_k|^2 \ge (1-\varepsilon^2) \sum_k |X_k|^2\;\).
  We implement this as cumulative energy over sorted spectral power.

Why allow nonzero ε? Compressibility is a sparsity–accuracy tradeoff. Larger ε (looser) permits smaller M (sparser) while maintaining a controlled error bound; smaller ε (tighter) forces larger M. This reflects real compute/memory budgets in FFT-based systems and ML models.

### Contributions (short)
- Arithmetic masks → spectral combs: periodic masks concentrate energy via frequency-domain convolution.
- Measurable gap: at fixed M, even < odd relative error; at fixed ε, odd needs slightly more modes; proportional M shows a stable small gap across L.
- Controls: unit-norm, DC removal, DCT/Hann – rankings persist; constants shift modestly.
- Asymptotics: fixed M with L→∞ ⇒ error→1; proportional M keeps error bounded and monotone in the fraction.
- Generalization: residue classes mod q and sweeps on the critical line; entropy/effective modes corroborate.

### Features
- Generalized ζ(s): real s>1 and complex s (critical line values supported; real part used for spectra).
- Residue classes: n ≡ r (mod q).
- Backends: NumPy by default; optional torch rFFT for GPU.
- Polars for dataframes; Matplotlib/Seaborn for plots.

### API (Python)
- `zeta_compress.zeta_sequence(L, s)` → array n^{-s}.
- `zeta_compress.even_mask(L)`, `odd_mask(L)`, `residue_mask(L,q,r)`.
- `zeta_compress.top_m_rel_error(seq, M, basis="rfft"|"dct", backend="numpy"|"torch")`.
- `zeta_compress.spectral_entropy(seq)` and `min_modes_for_error(seq, eps)`.
- `zeta_compress.analysis.ProportionalConfig`, `proportional_compressibility(cfg)`.

### Usage Examples
- Programmatic:
```python
import polars as pl
from zeta_compress import zeta_sequence, even_mask, odd_mask
from zeta_compress import top_m_rel_error

L=4096
base = zeta_sequence(L, 2.0)
seq_e = base * even_mask(L)
seq_o = base * odd_mask(L)
err_e = top_m_rel_error(seq_e, 64)
err_o = top_m_rel_error(seq_o, 64)
print(err_o/err_e)
```

### Results (empirical)
- For ζ(2), proportional runs consistently show odd subsequences require more modes than even for the same relative error (typically a few percent for the shown fractions). This gap persists across L and transforms (rFFT vs DCT-II), aligning with the Euler factor (1-2^{-s}). Fixed-ε tests (minimal M for target ε) reproduce the same ordering; at very small ε, both sequences saturate to nearly all bins so ratios approach 1.

### ML Applications
To be added with the experiments from Aug...

### Scope & Caveats
- We stay within Fourier-type compressibility of zeta-derived sequences under arithmetic masks. We do not claim complexity separations.
- Reported “few percent” gaps depend on ε/M, L, basis/window, and preprocessing; the sign and stability of the gap are robust, while exact numbers vary.
- Near machine-precision reconstructions, ratios can plateau/oscillate due to finite precision.
- Spectral features for Fourier Neural Operators and spectral transformers.
- Positional encodings from structured Dirichlet masks (even/odd or mod q) with controllable compressibility.
- Profiling shows modest overhead differences that map to layer FLOP budgets via O(n log n) scaling of FFTs.

### Robustness & Profiling
- Vary L and fractions; evaluate entropy and min modes.
- Optional GPU via torch: `top_m_rel_error(..., backend="torch", device="cuda")`.
- Add noise to sequences to test stability (extendable).

### Reproducibility
- Windows PowerShell runner: `uv run powershell -File scripts/run_all_tests.ps1`.
- Artifacts (CSVs/plots) are collected under `Test_Outputs/` with a summary at `Test_Outputs/test_run_summary.csv`.
- Legacy fixed-ε output: `Test_Outputs/legacy_min_modes_output.txt`.

#### Windowing and multi-taper (when to use)
- Default is fine: a single taper (Hann) or Tukey 0.25–0.5 typically stabilizes complex s runs.
- Use DPSS multi-taper (K>1) only if you observe instability: wide bootstrap CIs (> ~0.05 half-width) or large seed/window sensitivity (> ~2–3% changes).
- If needed, set window to `dpss:NW` with NW≈2.5–3 and K≈2NW−1; expect ~K× compute. Otherwise keep K=1 for speed.

### Contributing
See `CONTRIBUTING.md`. Use uv for environment and testing.

### Roadmap
- ζ(s) along the critical line with oscillatory terms; connect to zeros.
- Dirichlet characters and L-functions for residue-class filters.
- Confidence intervals over randomizations and noise models.
- Deeper ML examples (FNOs, transformer encodings) and GPU profiling with `torch.profiler`.

### to do ...
- Title: something pompous and grandiose for the LOLs ...
"Arithmetic Symmetry and Fourier Compressibility in Zeta Subsequences".
- Contributions: quantifying parity/residue-class compressibility gaps; linking Euler factors to spectral sparsity; ML implications for spectral architectures.

[To study other values, other than ζ(2) ASAP]

- ζ(s) along critical line; residue classes via Dirichlet characters.
- Torch GPU benchmarks and `torch.profiler` traces.
- Notebooks/plots illustrating compressibility gaps and ML uses.
etc. etc. etc.

### NOTES
- Setups that yield ≈7%:
  - Fixed-ε min-modes in the mid-accuracy regime (ε around 0.15–0.30), with DC kept, rFFT, no window, and L in the 4k–64k range. In that band M is far from “all bins,” so the parity structure shows up cleanly and integer rounding doesn’t collapse the gap. Repro:
    - PowerShell: `uv run python scripts/legacy_min_modes_runner.py --lengths 8192,32768 --eps 0.15,0.20,0.30`
  - Entropy/effective modes consistently show ~1.07 (odd/even) across L; that’s the most robust “7%” signal and aligns with the fixed-ε mid-range.

- When you won’t see it (ratios ≈ 1.0):
  - Very tight ε (e.g., 0.03–0.10 in our runs) forces keeping nearly all bins; M saturates and M_odd == M_even.
  - Very small L where integer steps dominate, or with aggressive windows/DC policies that push both to near-full coverage.

- Can the gap be larger?
  - Yes, in fixed-M error ratios at very tight budgets (small M/fractions) the odd/even error ratio can exceed 7% because both reconstructions are poor and small structural differences are amplified.
  - For fixed-ε bin ratios with parity masks, we generally see “few percent to ~10%” in the mid-range; larger gaps are more common with residue classes q>2 (not strictly parity).

- Why this is interesting
  - It’s a clean, reproducible link from arithmetic masks (Euler factor at p=2) to spectral sparsity and resource needs. It gives a concrete planning rule: for parity-masked ζ(2) subsequences, budget a few percent more modes (or fraction) for odd to match even at the same ε.
