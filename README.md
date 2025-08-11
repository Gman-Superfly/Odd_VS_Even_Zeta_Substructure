## Even vs Odd Zeta Subsequences: Concise README

### Scope
Neutral summary of the parity split of \(\sum 1/n^2\) and the FFT-based measurements implemented here. 

### Introduction
We experiment with compressable sequences, link arithmetic symmetry (e.g., the p=2 Euler factor) to algorithmic compressibility under Fourier-type bases. Using the even/odd subsequences of the Basel series 1/n^2 as a canonical case, we show that the even subsequence is consistently more FFT‑compressible than the odd subsequence under matched budgets, we show that the bins needed for odd are ~7% more than even.

### Mathematical basics
- \(\zeta(2) = \sum_{n\ge1} 1/n^2 = \pi^2/6\)
- Even terms: \(\sum_{k\ge1} 1/(2k)^2 = (1/4)\,\zeta(2) = \pi^2/24\)
- Odd terms: \(\sum_{k\ge1} 1/(2k-1)^2 = (1-2^{-2})\,\zeta(2) = \pi^2/8\)

### Dependencies
- Python 3.10+
- Required: numpy, mpmath
- Optional: scipy (for DCT-II; code falls back to rFFT if missing)

### Quick start 
From the project root:

```
python .\even_odd_zeta_fft.py
```

### What is measured
- Fixed-M compressibility: keep top \(M\) rFFT/DCT coefficients (by magnitude) of length-`L` sequences; report relative L2 reconstruction error.
- Proportional compressibility: choose \(M = \text{fraction} \times (L/2+1)\) and sweep fractions.
- Spectral metrics: entropy (base 2) and effective modes \(2^H\).
- Optional: mod-\(q\) residue-class masking, fixed-`M` vs length divergence, minimum modes to reach target \(\varepsilon\).

### Controls (implemented)
- Unit-norm comparison per length `L`.
- Optional DC removal (`remove_dc=True`).
- Optional Hann window (`window="hann"`).
- Optional DCT-II basis (`basis="dct"`; fallback to rFFT if unavailable).

### Options (edit function calls in code)
- Mean removal: `remove_dc=True`
- Windowing: `window="hann"`
- Basis: `basis="dct"`
- Proportional sweep: set `fraction` list
- Fixed-M vs length: set `M_list` and `L_powers`

### Outputs
- Console prints:
  - Convergence errors at selected term counts
  - Fixed-`M` compressibility errors and even/odd ratios
  - Spectral entropy `H` and effective modes `2^H`
  - Optional residue-class and fixed-`M` vs length results
- CSV logs (proportional runs): `results_compressibility_proportional_YYYYMMDD_HHMMSS.csv`
  - Columns: `L,fraction,M,even_rel_err,odd_rel_err,ratio_odd_over_even`

### Minimal findings (empirical)
- Convergence: even/odd have the same first-order rate; constants differ slightly.
- Compressibility: at matched `M` or fraction, even typically shows slightly lower relative error than odd. Ordering is stable across tested `L`, with and without DC.
- Fixed vs proportional: with fixed `M` and growing `L`, error increases toward 1; with proportional `M`, error is bounded and decreases with fraction.

### File map
- Code: `even_odd_zeta_fft.py`
- Concise study note: `EVEN_ODD_ZETA_SUBSEQUENCES.md`
- Full version (archived): `EVEN_ODD_ZETA_SUBSEQUENCES_full_fluff.md`
- CSV outputs: `results_compressibility_proportional_*.csv`


