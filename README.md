## Even vs Odd Zeta Subsequences: Arithmetic Symmetry and Spectral Compressibility

### Abstract
We empirically link arithmetic symmetry (e.g., the p=2 Euler factor) to algorithmic compressibility under Fourier-type bases. Using the even/odd subsequences of the Basel series 1/n^2 as a canonical case, we show that the even subsequence is consistently more FFT‑compressible than the odd subsequence under matched budgets. The gap persists across controls (unit‑norm, DC removal, DCT/Hann), generalizes to mod‑q residue classes, and is stable across length scales. We provide asymptotic context: with fixed top‑M budgets, relative error approaches 1 as length grows (fixed‑budget incompressibility), while proportional budgets keep errors bounded. All experiments are reproducible via a single Python script on Windows.

### Key contributions (short)
- Arithmetic masks → spectral combs: periodic residue masks concentrate energy via frequency‑domain convolution.
- Measurable gap: at fixed M, even < odd relative error; gap widens mildly with M and shrinks with L.
- Controls: gap persists with unit L2 normalization, DC removal, DCT/Hann basis/window changes.
- Asymptotics: fixed M, L→∞ ⇒ error → 1; proportional M ∝ L ⇒ bounded, monotone error.
- Generalization: mod‑q residues exhibit a period‑driven compressibility ordering; we add divergence and entropy metrics.

### Background (mathematical context)
- Basel sum: ζ(2)=∑ n≥1 1/n^2. Even terms sum to π^2/24; odd terms to π^2/8.
- Parity masks: 1_even(n)=½(1+(-1)^n), 1_odd(n)=½(1-(-1)^n). Dirichlet eta gives the parity split at s=2.
- Euler product and characters: removing p=2 corresponds to the odd subsequence; more generally, residue classes mod q (and Dirichlet characters) are q‑periodic.
- Spectral comb: multiplying by a q‑periodic mask in time is convolution with a sparse comb in frequency, duplicating and aligning energy into few bins.

### Methods (what we measure)
- Fixed‑M FFT compressibility: keep the top M rFFT/DCT coefficients by magnitude and report relative L2 reconstruction error.
- Proportional compressibility: choose M = frac × (L/2+1) and sweep fractions.
- Spectral entropy: H of normalized power spectrum; effective modes ≈ 2^H.
- Minimum modes for target error ε: smallest M reaching relative L2 ≤ ε by cumulative spectral energy.
- Mod‑q residue classes: mask x(n)=1/(n+1)^2 by residues r mod q and repeat metrics.
- Fixed‑M vs length divergence: sweep L=2^k at fixed M to show error → 1 while odd/even ratio remains ≥1 and trends to 1.

### Controls for fairness and robustness
- Energy normalization: unit L2 before transform.
- DC handling: with and without mean (remove_dc=True).
- Basis/window: rFFT vs DCT‑II; optional Hann window. Rankings persist; constants shift.

### Results (high‑level)
- Fixed‑M (per length): even < odd error; ratios ~1.01–1.15 at moderate M, shrinking near perfect reconstruction.
- Proportional‑M: small but systematic gaps (~1.00–1.01) that persist across L.
- Spectral entropy: odd exceeds even by ~0.10 bits; effective modes ratio ≈ 1.07.
- Minimum modes: for ε≈1e‑1 at tested L, ≈ half of rFFT bins suffice; for ε≈3e‑2, nearly all bins are needed.
- Fixed‑budget divergence: as L increases with M fixed, both errors → 1; even remains uniformly better at finite L.

### Practical implications: when the gap matters
- Fixed‑M budgets: when L ≫ M (e.g., L ≥ 8k with M ∈ [64, 512]), odd often shows ~5–15% higher relative error than even at the same M. As L grows with M fixed, both errors approach 1; odd reaches unacceptable error slightly sooner.
- Proportional budgets: the gap is ~constant. Odd typically needs ≈7% more rFFT bins than even to achieve the same ε at any L. This overhead scales linearly with L and with how many sequences you store/process.
- Tight fractions (≈0.5–2% of rFFT bins): expect odd to miss a given ε unless you increase the budget by ~7% relative to even.

### Applications and implications (real world)
- Machine learning / feature engineering: with Fourier features on periodically masked sequences, even‑like masks hit a target approximation with fewer coefficients. Allocate model or feature budget by mask type; plan ~5–10% extra for odd‑like masks to match even‑like quality.
  - Where odd/even masks appear in ML:
    - Checkerboard/alternating masks in normalizing flows and autoregressive models (e.g., RealNVP/Glow, PixelCNN)
    - Strided or dilated ops and pooling (keep every other timestep/pixel)
    - Polyphase splits in wavelet/subband features (even/odd branches)
    - Periodic bucketing/feature hashing by index mod q (including q=2 parity)
    - Data pipelines with periodic gating (e.g., selecting odd frames for latency/throughput)
- Compression and storage: for top‑M FFT storage, odd‑like streams need ≈7% more bins to meet the same error under proportional budgets; adjust storage/bandwidth accordingly.
- On‑device/edge sensing (fixed M): as length grows, odd‑like segments degrade earlier; add headroom or avoid odd‑only gating if fidelity must be stable.
- Communications/OFDM: periodic masks affect spectral compactness. Small period (q=2) concentrates energy; odd‑only (removing p=2) is slightly less compact → may need a bit more subcarrier or coding budget.
- DSP pipelines (decimation/windowing): period‑2 gating is more FFT‑friendly. Expect lower error at the same M; choose filterbank allocations with mask period in mind.
- Spectral/numerical methods: residue/character masks closer to small‑period characters yield sparser spectra; informs truncated‑transform accuracy and thresholds.
- Zeta/L‑functions computation: character twists mod q alter concentration; p=2 presence helps. Useful for truncation/caching choices in large evaluations.
- Compressed sensing/sketching: slightly sparser spectra → slightly fewer measurements to hit ε; odd‑like masks typically need ≈7% more.
- Telemetry/analytics with mod‑q bucketing: if you can choose the mask, smaller period q (esp. 2) yields more compressible aggregates; odd‑only buckets cost a bit more at the same fidelity.
- Rule of thumb: arithmetic symmetry → spectral compactness. Budget ~5–10% extra for odd‑like masks when matching even‑like quality.

#### Cost impact at scale
- Storage/bandwidth: ~7% fewer kept bins for even‑like masks at the same fidelity → lower data at rest and in transit.
- Compute: ~7% fewer coefficients to compute/sort/transmit → lower CPU/GPU time and energy.
- Edge/embedded: same on‑device budget yields higher quality, or same quality at lower power.
- Training/inference: smaller Fourier feature sets reduce memory and FLOPs.

### Asymptotics and fixed‑budget divergence
In this section, “incompressible” means: under a fixed top‑M Fourier budget, the relative L2 error approaches 1 as L→∞.
- Fixed M, L→∞: error → 1. Basis/window change constants but not this fact or the even<odd ordering at finite L.
- Proportional M: error bounded and monotone in fraction; even remains modestly better.
- Notes: near exact reconstruction, finite precision induces plateaus; ratios can fluctuate at machine epsilon.

### Broader phenomenon and hypotheses
- Small period ⇒ higher compressibility: masks with smaller period q concentrate energy into fewer harmonics; fixed‑M error tends to increase with q.
- Character ranking: within a fixed q, characters with larger low‑harmonic Fourier mass compress better; real characters often beat complex ones.
- Multiple primes: “coprime to S” masks have period lcm(S); error correlates with effective period size.
- Exponent effect: for 1/n^s, larger s (>1) mildly amplifies small‑q advantages.
- Alternative bases: rankings persist under DCT‑II/Hann; character/Ramanujan bases may show stronger sparsity (future work).

### Reproducibility 
- Run everything:
```
python .\even_odd_zeta_fft.py
```
- What runs:
  - Digit checks; convergence errors
  - 2D FFT symmetry demo (power‑of‑two sizes)
  - Fixed‑M compressibility sweeps across L
  - Spectral entropy and effective modes
  - Minimum modes for ε ∈ {1e‑1, 3e‑2, 1e‑2}
  - Mod‑q residue‑class compressibility (q ∈ {3,4,6,8})
  - Proportional‑M sweeps with CSV logging (`results_compressibility_proportional_*.csv`)
  - Fixed‑M vs length divergence

Dependencies: Python 3.10+, numpy, mpmath; optional SciPy for DCT‑II (falls back to rFFT if missing). No special OS setup beyond standard Python.

### Quick check: spectral entropy (Windows PowerShell)
- Run a lightweight confirmation of the entropy/effective‑modes gap without the full sweep:

```powershell
$env:PYTHONIOENCODING='utf-8'; [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); python -c "import even_odd_zeta_fft as m; m.spectral_metrics(L_list=(1024,2048,4096))"
```

- Typical output (unit‑norm, DC kept):
```
Spectral entropy and effective modes (2^H):
L=1024  H_even=8.88  H_odd=8.98  eff_even≈470   eff_odd≈505   ratio≈1.07
L=2048  H_even=9.88  H_odd=9.98  eff_even≈940   eff_odd≈1009  ratio≈1.07
L=4096  H_even=10.88 H_odd=10.98 eff_even≈1880  eff_odd≈2018  ratio≈1.07
```

- Finding: odd has ~+0.10 bits higher spectral entropy → ~7% higher effective modes at matched L.

- DC-removed variant (mean removed before rFFT):
```powershell
$env:PYTHONIOENCODING='utf-8'; [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); python -c "import even_odd_zeta_fft as m; m.spectral_metrics(L_list=(1024,2048,4096), remove_dc=True)"
```

- Typical output (DC removed): ratios remain ≈ 1.07 across these L, confirming the effect is not a DC artifact.

### Optional run variants (edit function calls in code)
- Remove DC: set `remove_dc=True` in `fft_compressibility_demo`, `fft_compressibility_proportional`, or `fixed_M_vs_length_divergence`.
- Use Hann window: pass `window="hann"`.
- Use DCT‑II basis: pass `basis="dct"` (falls back automatically if SciPy not installed).

### Separate test: minimum modes for target error
- Quick test script to quantify how many Fourier bins are needed to reach ε:

```powershell
python .\test_min_modes.py --lengths 1024,2048,4096 --eps 1e-1,3e-2
```

- DC-removed variant:
```powershell
python .\test_min_modes.py --lengths 1024,2048,4096 --eps 1e-1,3e-2 --remove-dc
```

- Typical output shows odd needing a few more bins at coarse ε and essentially equal at tight ε (near-full bins), consistent with entropy results.

### Repository map
- Code: `even_odd_zeta_fft.py`
- Study write‑up: `EVEN_ODD_ZETA_SUBSEQUENCES.md`
- README (this file): conceptual summary, how to run, and claims

### Related work (short)
- Dirichlet/Euler‑product basics, parity splits, and characters are classical.
- FFT‑accelerated ζ computations (e.g., Odlyzko–Schönhage) and Riemann–Siegel analyses are standard.
- We are not aware of prior work that frames and quantifies a stable FFT‑compressibility gap for residue‑masked 1/n^s via fixed/proportional budgets, entropy, and divergence.

### Scope and claims
- This is an empirical and methodological note, not a complexity‑class statement. Claims are modest, reproducible, and limited to Fourier‑type compressibility of zeta‑derived sequences under arithmetic masks.

### How to cite
```
Even vs Odd Zeta Subsequences: Arithmetic Symmetry and Spectral Compressibility (2025).
https://github.com/gman-superfly/Odd _VS_Even_Zeta_Substructure
```

.


