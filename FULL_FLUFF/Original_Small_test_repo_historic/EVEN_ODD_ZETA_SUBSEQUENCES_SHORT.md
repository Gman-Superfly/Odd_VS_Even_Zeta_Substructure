## Even vs Odd Zeta Subsequences: Concise Summary and Fun Explorations

### Scope
Brief, neutral summary of the parity split of \(\sum 1/n^2\) and the FFT-based measurements implemented in `even_odd_zeta_fft.py`. Focus: what is computed, how to run, what is output.

### Mathematical basics
- \(\zeta(2) = \sum_{n\ge1} 1/n^2 = \pi^2/6\)
- Even terms: \(\sum_{k\ge1} 1/(2k)^2 = (1/4)\,\zeta(2) = \pi^2/24\)
- Odd terms: \(\sum_{k\ge1} 1/(2k-1)^2 = (1-2^{-2})\,\zeta(2) = \pi^2/8\)
- Parity masks via Dirichlet eta \(\eta(s)=(1-2^{1-s})\zeta(s)\):
  - \(\sum \mathbf{1}_{\text{even}}(n)/n^s = \tfrac12\zeta(s) - \tfrac12\eta(s)\)
  - \(\sum \mathbf{1}_{\text{odd}}(n)/n^s  = \tfrac12\zeta(s) + \tfrac12\eta(s)\)

### What the code measures
- Convergence comparison: partial-sum errors (even vs odd) at matched term counts.
- FFT compressibility: keep top \(M\) rFFT modes of length-`L` sequences; report relative L2 reconstruction error.
- Spectral metrics: entropy (base 2) and effective modes \(2^H\).
- Optional: residue-class (mod-\(q\)) variants, minimum modes to hit target error \(\varepsilon\), proportional-\(M\) sweeps.

### Controls (implemented)
- Unit-norm comparison per length `L`.
- Optional DC removal (`remove_dc=True`).
- Optional Hann window (`window="hann"`).
- Optional DCT-II basis (`basis="dct"`, falls back to rFFT if SciPy missing).

### How to run 
From the project root:

```
python .\even_odd_zeta_fft.py
```


### Key outputs
- Console prints:
  - Convergence errors at selected term counts.
  - Fixed-`M` compressibility errors for even/odd and their ratio.
  - Spectral entropy `H` and effective modes `2^H` by `L` (optionally with DC removed).
  - Optional residue-class results and fixed-`M` vs length sweeps.
- CSV logs (proportional compressibility): files named `results_compressibility_proportional_YYYYMMDD_HHMMSS.csv` with columns
  `L,fraction,M,even_rel_err,odd_rel_err,ratio_odd_over_even`.

### Options (in-code flags)
Examples (edit calls in `even_odd_zeta_fft.py` as needed):
- Mean removal: `remove_dc=True`
- Windowing: `window="hann"`
- Basis: `basis="dct"`
- Proportional budget: set `fraction` values in the sweep helper

### Minimal interpretation
- Convergence: even and odd have the same first-order asymptotics; constants differ modestly.
- Compressibility (empirical): at matched `M` or fraction, the even sequence typically shows slightly lower relative error than the odd sequence. The ordering is stable across tested lengths and with/without DC.
- Fixed vs proportional budgets: with fixed `M` and increasing `L`, relative error increases toward 1; with proportional `M`, errors are bounded and decrease with the kept fraction.

### File map
- Code: `even_odd_zeta_fft.py`
- Doc (this): `EVEN_ODD_ZETA_SUBSEQUENCES.md`
- CSV outputs: `results_compressibility_proportional_*.csv`
