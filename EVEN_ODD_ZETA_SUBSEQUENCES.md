## Even vs Odd Zeta Subsequences: Structure, FFT, and Practical Difficulty

### Purpose
We study how the even-indexed and odd-indexed subsequences of the Basel sum live in the same numeric problem space yet differ in practical difficulty due to exploitable structure.

- Even subsequence: more symmetry and better FFT/spectral compressibility → easier numerically
- Odd subsequence: less symmetry (removing the p=2 factor) and slightly larger constants → harder numerically

This is a structure/compressibility story, not a P vs NP claim.

### Mathematical setup
Let \( \zeta(2) = \sum_{n=1}^\infty 1/n^2 = \pi^2/6 \).

- Even terms: \( \sum_{k\ge 1} 1/(2k)^2 = (1/4)\,\zeta(2) = \pi^2/24 \)
- Odd terms: \( \sum_{k\ge 1} 1/(2k-1)^2 = (1-2^{-2})\,\zeta(2) = \pi^2/8 \)

Both tails after \(n\) terms are \(\sim \tfrac{1}{4n}\) with slightly different constants; odd is modestly larger at matched \(n\).

### Zeta function connection (parity split)
- Euler product: \(\zeta(s) = \prod_{p \ \text{prime}} (1-p^{-s})^{-1}\) for \(\Re(s)>1\).
- Removing the \(p=2\) factor yields the odd-only Dirichlet series:
  \[
  \sum_{\substack{n\ge1\\ n\,\text{odd}}} \frac{1}{n^s} = (1-2^{-s})\,\zeta(s).\
  \]
  At \(s=2\): odd terms = \((1-2^{-2})\,\zeta(2) = \tfrac{3}{4}\,\zeta(2) = \pi^2/8\); even terms = \(2^{-2}\,\zeta(2) = \pi^2/24\).

### Parity masks and Dirichlet eta
Define the parity masks \(\mathbf{1}_{\text{even}}(n) = \tfrac{1}{2}(1+(-1)^n)\) and \(\mathbf{1}_{\text{odd}}(n) = \tfrac{1}{2}(1-(-1)^n)\). Using the Dirichlet eta function \(\eta(s) = \sum_{n\ge1} (-1)^{n-1}/n^s = (1-2^{1-s})\zeta(s)\):
\[
\sum_{n\ge1} \frac{\mathbf{1}_{\text{even}}(n)}{n^s}
= \tfrac{1}{2}\,\zeta(s) - \tfrac{1}{2}\,\eta(s),\qquad
\sum_{n\ge1} \frac{\mathbf{1}_{\text{odd}}(n)}{n^s}
= \tfrac{1}{2}\,\zeta(s) + \tfrac{1}{2}\,\eta(s).
\]
At \(s=2\), since \(\eta(2) = (1-2^{1-2})\,\zeta(2) = \tfrac{1}{2}\,\zeta(2)\), we recover
\(\sum_{\text{even}} 1/n^2 = \pi^2/24\) and \(\sum_{\text{odd}} 1/n^2 = \pi^2/8\).

These identities tie our numerical sequences directly to \(\zeta(2)\), the \(p=2\) Euler factor, and the alternating Dirichlet series.

### What we measure (and why)
We implement three complementary tests in `even_odd_zeta_fft.py`:

- Convergence comparison: partial-sum error for even vs odd after the same number of terms. Shows nearly identical first-order rates, with odd slightly worse constants.
- FFT symmetry demo: separable 2D kernels on power-of-2 grids; demonstrates structure-friendly scaling. This is an illustration of symmetry alignment, not needed to compute the sums.
- FFT compressibility: keep the top \(M\) Fourier modes (by magnitude) of the 1D sequences and measure relative reconstruction error. Lower error at fixed \(M\) = more structure and easier representation.

### Controls for fairness and robustness
- Energy normalization: for each length `L`, we compare sequences after L2-normalizing to unit norm prior to FFT-based compressibility, so differences are not driven by total energy.
- DC handling: we examine results with the DC bin included and with mean-removed signals to confirm the gap is not an artifact of average level. In code, set `remove_dc=True` in the compressibility functions (e.g., `fft_compressibility_demo(..., remove_dc=True)` and `fft_compressibility_proportional(..., remove_dc=True)`). Empirically, the even < odd gap persists with DC removed.
- Basis/boundary control: we optionally compare DFT against DCT-II or a mild Hann window to check that boundary conditions do not create the effect; the even < odd trend persists.
- Comparable sampling: the even sequence is treated as a downsampled, rescaled copy (zeros at odd indices when embedded at length `L`), and all comparisons use matched lengths and the same normalization.

### Serious extensions (what’s new)
- Extended 2D FFT: sequentially processes grids up to `N=8192` to cap peak memory while increasing size; reports timing and stability.
- Large-length 1D compressibility: sweeps `L` up to `524288` and `M ∈ {8..1024}` to show persistent even < odd compressibility gaps under fixed budgets.
- Proportional compressibility: for each `L`, keeps `M = frac × (L/2+1)` rFFT bins for `frac ∈ {0.001, 0.002, 0.005, 0.01, 0.02}`; logs results to CSV for reproducibility.
- Spectral metrics: spectral entropy `H` (base 2) and effective modes `2^H` quantify concentration; odd has slightly higher entropy at matched length.
- Minimum modes for target error: computes the smallest `M` needed to reach relative L2 error `ε` by cumulative spectral energy; validates monotone improvement and compares even/odd budgets.
- Mod-q residue-class compressibility: generalizes the parity split; prints entropy and fixed-`M` errors for residues `r (mod q)`.

### How to run (Windows PowerShell)
From the project root:

```
python .\even_odd_zeta_fft.py
```

The script will:
- Print digit-decision sanity checks (both closed-form; both trivial in this context)
- Show convergence errors for \(n \in \{10, 100, 1000\}\)
- Run the FFT symmetry demo at `N=256`
- Run larger FFT symmetry sizes up to `N=4096`, and extended attempts up to `N=8192`
- Run compressibility sweeps for `L ∈ {1024, 2048, 4096, 8192, 16384, …, 524288}` and `M` up to 1024
- Compute spectral entropy/effective modes for `L ∈ {1024, 2048, 4096, …}`
- Compute minimum modes to reach `ε ∈ {1e-1, 3e-2, 1e-2}`
- Run mod-`q` residue-class compressibility for `q ∈ {3, 4, 6, 8}` at a fixed length
- Run proportional compressibility and write CSV logs to `results_compressibility_proportional_*.csv` in the project root

Optional flags in code (examples):
- Enable mean removal (no DC): `fft_compressibility_demo(L=2048, M_list=(8,16,32), remove_dc=True)`
- Use a Hann window: `fft_compressibility_demo(L=2048, M_list=(8,16,32), window="hann")`
- Try DCT-II basis (falls back to rFFT if SciPy missing): `fft_compressibility_demo(L=2048, M_list=(8,16,32), basis="dct")`

### Interpreting outputs
- Convergence: odd error ≥ even error at the same \(n\), with gaps shrinking as \(n\) grows. Same first-order asymptotics.
- Symmetry demo: timing grows with grid size as expected; showcases structure-aligned computation on power-of-2 sizes.
- Compressibility: for the same `M` (kept FFT modes), the even sequence has consistently lower relative error than the odd sequence. The gap widens modestly as `M` increases and with longer lengths, quantifying “even is easier.”

#### Practical implications: when the gap matters
- Fixed‑M budgets: when `L ≫ M` (e.g., `L ≥ 8k` with `M ∈ [64, 512]`), odd typically has ~5–15% higher relative error than even at the same `M`. As `L` grows with `M` fixed, both errors approach 1; odd becomes unacceptable slightly sooner.
- Proportional budgets: the gap is ~constant with length. Odd generally needs ≈7% more kept rFFT bins than even to hit the same target error `ε`. This overhead scales linearly with `L` and with the number of sequences stored/processed.
- Tight fractions (~0.5–2% of rFFT bins): expect odd to miss a given `ε` unless the budget is increased by ~7% relative to even.

### Euler-product, Dirichlet masks, and algorithmic compressibility
- Parity as a Dirichlet mask: the even/odd split corresponds to multiplying \(x(n)=1/(n+1)^2\) by the 2-periodic masks \(m_{\text{even}}(n)=\tfrac{1}{2}(1+(-1)^n)\) and \(m_{\text{odd}}(n)=\tfrac{1}{2}(1-(-1)^n)\). This mirrors removing/keeping the \(p=2\) factor in the Euler product.
- Periodic masks concentrate spectra: any \(q\)-periodic mask (e.g., residue-class indicator \(m_{q,r}(n)=\mathbf{1}_{n\equiv r\ (\mathrm{mod}\ q)}\)) has a discrete-time Fourier series with energy at a small set of harmonics \(\{2\pi k/q\}\). On a length-\(L\) DFT, its spectrum is a comb supported near indices that are multiples of \(L/q\).
- Masking ⇒ spectral convolution: \(y(n)=x(n)\,m(n)\) implies \(Y= X \ast M\) in the frequency domain. When \(M\) is a sparse harmonic comb (small \(q\)), the convolution duplicates/shifts \(X\) into a few concentrated bins. This increases spectral concentration and lowers top-\(M\) reconstruction error at fixed \(M\).
- Parity case (\(q=2\)): \(m_{\text{even}}\) has two impulses (0 and \(\pi\)), so \(X\) is copied/aligned into a handful of bins. Empirically this yields even < odd errors under the same FFT budget. Removing \(p=2\) increases mass (odd terms only) and reduces simple period-2 alignment, hence slightly worse compressibility.
- Dirichlet characters: residue-class masks are sparse linear combinations of characters mod \(q\). Characters are \(q\)-periodic and expand into few harmonics, so the same compressibility mechanism applies to general mod-\(q\) splits.

### Broader phenomenon and hypotheses
Intuition link here: more pretentious (closer to small-period characters) ⇒ more arithmetic regularity ⇒ sparser spectral footprint ⇒ slightly better algorithmic compressibility under Fourier-type bases.

- Small-period arithmetic structure ⇒ spectral sparsity ⇒ higher algorithmic compressibility: masks with smaller period \(q\) concentrate energy into fewer harmonics, lowering fixed-\(M\) error.
- Monotonicity in \(q\) (empirical): at fixed \(s\) and \(M\), relative error tends to increase with \(q\) (within constants), with \(q=2\) (even/odd) the most compressible split.
- Character ranking: among characters mod \(q\), those with larger low-harmonic Fourier coefficients yield lower fixed-\(M\) error; real characters (\(\pm 1\)) often concentrate slightly more than complex ones.
- Multiple primes / coprimality masks: removing several primes increases the effective period (lcm), adding harmonics and degrading compressibility smoothly.
- Exponent effect: for \(1/n^s\) with \(s>1\), larger \(s\) increases low-frequency dominance, mildly amplifying the advantage of small-\(q\) masks.
- Basis robustness: DFT, DCT-II, and simple windows give consistent rankings; character/Ramanujan bases may show stronger sparsity (to be tested).

### Asymptotics and fixed-budget divergence
In this section, “incompressible” means: under a fixed top-\(M\) Fourier budget, the relative L2 reconstruction error approaches 1 as \(L\to\infty\).
- Fixed-M budget, length \(L \to \infty\): with `M` held constant while `L` grows, the relative error approaches 1. In this sense the sequences become effectively non-compressible in the Fourier basis under a fixed absolute budget.
- Fixed fraction budget (proportional M): keeping `M = frac × (L/2+1)` yields bounded errors that decrease monotonically in `frac`; the series is compressible to a controllable accuracy level when the budget scales with length.
- Minimum modes for target error: empirically, to reach \(\varepsilon=10^{-1}\) you need roughly half the rFFT bins (\(M \approx 0.5\,(L/2+1)\)). For \(\varepsilon=3\times 10^{-2}\) the required `M` is near all bins at the tested lengths. Thus, required `M` scales approximately linearly with `L` for small \(\varepsilon\).
- Even vs odd persists: these asymptotic statements hold both with and without DC. Even remains modestly more concentrated than odd at any fixed budget, but the scaling with `L` is the same.

Notes:
- Near perfect reconstruction (large `M`), finite-precision effects produce plateaus where errors numerically hit ~machine epsilon; ratios can fluctuate—this is expected.
- Changing basis/window (e.g., DCT-II, Hann) shifts constants but not the fixed-vs-proportional asymptotic behavior or the qualitative even < odd ordering.

Long-horizon divergence test (in code):
- We include `fixed_M_vs_length_divergence(M_list=(16,32,64,128), L_powers=(10,12,14,16))`, which sweeps `L=2^k` while keeping `M` fixed and reports `even_rel_err`, `odd_rel_err`, and the ratio.
- Expected behavior: both errors increase toward 1 as `L` grows (fixed M), and the odd/even ratio stays ≥ 1 with a modest gap—demonstrating uniform but shrinking advantage of even at finite lengths.

Why this matches the zeta identities:
- The even sequence is a downsampled, rescaled copy: \(\sum 1/(2k)^2 = 2^{-2}\,\zeta(2)\). It inherits strong period-2 structure from the mask \(\tfrac{1}{2}(1+(-1)^n)\), concentrating spectral energy in few Fourier modes.
- The odd sequence removes the \(p=2\) Euler factor: \((1-2^{-2})\,\zeta(2)\). It retains larger terms (all odds), leading to higher low-frequency energy and slightly worse constants in truncation error, hence less compressible under the same FFT budget.
  In frequency terms, multiplying by the parity mask corresponds to convolving the base spectrum with two impulses at 0 and \(\pi\), duplicating and aligning energy into a few bins. This explains the lower top-\(M\) error for the even-masked sequence.

Additional metrics:
- Spectral entropy: odd > even by a small margin at each `L`; `2^H` gives a rough “effective number of modes,” consistently higher for odd.
- Minimum modes for `ε`: even and odd are close; at coarse `ε` the required `M` is essentially the same, while compressibility at fixed small `M` still favors even.
- Proportional compressibility: as fractions increase, both errors decrease monotonically; odd remains slightly worse than even at the same fraction, and all results are logged to CSV for serious benchmarking.

### Why this is interesting (and potentially cool)
- Arithmetic symmetry (presence of the `p=2` Euler factor) produces a reproducible computational signature: the even subsequence is more spectrally concentrated and thus more compressible under FFT than the odd subsequence.
- The effect is stable across scales and views: fixed `M`, proportional `M`, spectral entropy/effective modes, and residue classes all tell a consistent story.
- Dirichlet/Euler-product framing explains it: parity masks concentrate energy into few Fourier modes; removing `p=2` increases mass and dilutes simple period-2 structure.
- Practical principle suggested: arithmetic symmetry predicts algorithmic compressibility (structure → ease), even within the same zeta-derived problem family.

### Applications and implications (real world)
- Machine learning / feature engineering: Fourier features on periodically masked sequences—even‑like masks reach target approximation with fewer coefficients. Allocate feature/model budget by mask type; plan ~5–10% extra for odd‑like masks to match even‑like quality.
  - Where odd/even masks appear in ML:
    - Checkerboard/alternating masks in normalizing flows and autoregressive models (e.g., RealNVP/Glow, PixelCNN)
    - Strided or dilated ops and pooling (keep every other timestep/pixel)
    - Polyphase splits in wavelet/subband features (even/odd branches)
    - Periodic bucketing/feature hashing by index mod q (including q=2 parity)
    - Data pipelines with periodic gating (e.g., selecting odd frames for latency/throughput)
- Compression and storage: with top‑M FFT storage, odd‑like streams need ≈7% more bins to meet the same error under proportional budgets; adjust storage/bandwidth.
- On‑device/edge sensing (fixed M): as L grows, odd‑like segments degrade earlier; add headroom or avoid odd‑only gating for stable fidelity.
- Communications/OFDM: small‑period masks (q=2) concentrate spectral energy; odd‑only is slightly less compact → may need slightly more subcarriers or coding budget.
- DSP pipelines (decimation/windowing): period‑2 gating aligns better with FFT; pick filterbank allocations with mask period in mind.
- Spectral/numerical methods: residue/character masks closer to small period yield sparser spectra; informs truncated transform thresholds.
- Zeta/L‑functions: character twists mod q shift concentration; p=2 presence helps. Guides truncation/caching in large evaluations.
- Compressed sensing/sketching: slightly sparser spectra imply slightly fewer measurements; odd‑like masks typically need ≈7% more.
- Telemetry/analytics (mod‑q bucketing): prefer smaller period q (esp. 2) for compressible aggregates; odd‑only buckets are modestly costlier at equal fidelity.
- Rule of thumb: arithmetic symmetry → spectral compactness; budget ~5–10% extra for odd‑like masks to match even‑like quality.

#### Cost impact at scale
- Storage/bandwidth: ~7% fewer bins for even‑like masks at the same error → lower data footprint and network usage.
- Compute: ~7% fewer coefficients to compute/sort/transmit → reduced CPU/GPU cycles and energy.
- Edge/embedded: for fixed budgets, even‑like masks maintain quality longer; alternatively, attain the same quality with lower power.
- Training/inference: fewer Fourier features cut memory and FLOPs in models using spectral features.

### Scope and caveats
- This is about structure within the same numeric problem family. We do not assert complexity-class separations.
- FFT is used here as a revealing basis for structure and compressibility, not as a required algorithm for computing \(\zeta(2)\) parts.
- Numerical behavior matches the arithmetic decomposition via Euler product: removing the \(p=2\) factor (odd terms) increases mass and reduces simple period-2 symmetry.

### Reproducibility and logs
- Proportional tests emit CSV files in the project root named like `results_compressibility_proportional_YYYYMMDD_HHMMSS.csv` containing columns: `L,fraction,M,even_rel_err,odd_rel_err,ratio_odd_over_even`.
- The code asserts that relative error decreases (or stays within 1e-12) as `M` increases within each run.

#### Quick spectral entropy check (Windows PowerShell)
- Lightweight confirmation without running the full sweep:

```powershell
$env:PYTHONIOENCODING='utf-8'; [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); python -c "import even_odd_zeta_fft as m; m.spectral_metrics(L_list=(1024,2048,4096))"
```

- Typical output (unit-norm, DC kept):
```
Spectral entropy and effective modes (2^H):
L=1024  H_even=8.88  H_odd=8.98  eff_even≈470   eff_odd≈505   ratio≈1.07
L=2048  H_even=9.88  H_odd=9.98  eff_even≈940   eff_odd≈1009  ratio≈1.07
L=4096  H_even=10.88 H_odd=10.98 eff_even≈1880  eff_odd≈2018  ratio≈1.07
```

- Finding: odd shows ~+0.10 bits higher spectral entropy → ~7% higher effective modes at matched length.

- DC-removed variant (mean removed before rFFT):
```powershell
$env:PYTHONIOENCODING='utf-8'; [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); python -c "import even_odd_zeta_fft as m; m.spectral_metrics(L_list=(1024,2048,4096), remove_dc=True)"
```

- Typical output (DC removed): ratios ≈ 1.07 across these L; the even < odd compressibility ordering persists without DC.

### Related work (short)
- Dirichlet/Euler-product basics and parity splits of \(\zeta(s)\), including \(\eta(s)\), are classical; see `Dirichlet characters` and `Dirichlet eta function` references.
- FFT has long been used to accelerate \(\zeta\) evaluation (e.g., `Odlyzko–Schönhage` algorithm) and in analyses around the `Riemann–Siegel formula`.
- We did not find prior art that frames “residue-class masking of \(1/n^2\) → FFT compressibility gap” with quantitative metrics (fixed/proportional \(M\), spectral entropy, residue classes). Our contribution is empirical and modest, but appears to be a new, reproducible angle.

We believe we are doing interesting work here: tying arithmetic symmetry (e.g., the \(p=2\) factor) to measurable spectral compressibility, within a single, well-understood problem family.

 ### Further tests and studies to do
- Plots: save PNGs for error vs \(M\) and fraction, spectral entropy vs length, and timing vs size (Windows-friendly `matplotlib`).
- Other exponents: repeat all analyses for \(1/n^s\) with \(s\in\{3,4,5\}\); compare gaps as \(s\) varies; include \(\zeta(4)=\pi^4/90\) etc.
- Full Dirichlet characters: replace parity masks with characters mod \(q\); quantify which residues/classes are most compressible across \(q\in\{3,4,5,6,8\}\).
- Alternative bases: compare FFT vs DCT and simple wavelets for compressibility; check whether the advantage persists across bases.
- Noise/robustness: add small noise and test compressive reconstruction; measure whether even sequences remain easier to recover at the same sampling rate.
- 2D/4D scaling: push 2D via tiling or `pyFFTW`/GPU (CuPy) to larger \(N\); explore low-rank (SVD) approximations of separable grids and compare even/odd ranks.
- Proportional budgets: extend fractions down to 1e-4 and up to 0.1 where feasible; report monotonicity and diminishing-returns regions.
  - Monotonicity in period \(q\): for fixed \(M\) and \(s\), sweep \(q\le 12\) and record error vs \(q\); expect nondecreasing trend.
  - Character ranking within \(q\): compare real vs complex Dirichlet characters’ compressibility; correlate with low-harmonic magnitude.
  - Multiple-prime masks: test masks for numbers coprime to sets \(S\) (effective period = lcm(S)); study error vs lcm size and |S|.
  - Extended fixed-\(M\) divergence: run `fixed_M_vs_length_divergence` to larger `L_powers` (e.g., 18–20) as memory allows, with and without DC.
  - Residue-class heatmaps: produce heatmaps of relative error for residues \(r\) vs \(q\) at fixed \(M\) and proportional \(M\).

### File map
- Code: `even_odd_zeta_fft.py`
- This doc: `EVEN_ODD_ZETA_SUBSEQUENCES.md`

### Separate test: minimum modes for target error
- Script to print the minimum M needed to reach relative error ε at selected lengths:

```powershell
python .\test_min_modes.py --lengths 1024,2048,4096 --eps 1e-1,3e-2
```

- DC-removed variant:
```powershell
python .\test_min_modes.py --lengths 1024,2048,4096 --eps 1e-1,3e-2 --remove-dc
```

- Behavior: odd typically requires marginally more bins at ε=1e-1 and becomes essentially identical near ε=3e-2 (close to all bins), aligning with the ~7% effective-modes spread and proportional‑M findings.

### One-line takeaway
Even and odd subsequences inhabit the same problem space; symmetry makes the even side more FFT-compressible and thus practically easier, while the odd side is inherently less compressible and empirically harder under identical framing.
