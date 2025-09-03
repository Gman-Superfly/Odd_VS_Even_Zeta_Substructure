from __future__ import annotations

import argparse
import math
import numpy as np
import polars as pl

from zeta_compress.analysis import ProportionalConfig, proportional_compressibility


def bootstrap_ratio_ci(values: np.ndarray, *, alpha: float = 0.05) -> tuple[float, float, float]:
	assert values.size > 1
	lower = np.quantile(values, alpha / 2)
	upper = np.quantile(values, 1 - alpha / 2)
	return float(np.mean(values)), float(lower), float(upper)


def main() -> int:
	p = argparse.ArgumentParser(description="Bootstrap CI for odd/even error ratio across resamples")
	p.add_argument("--L", type=str, default="4096,16384,65536")
	p.add_argument("--fractions", type=str, default="0.002,0.005")
	p.add_argument("--B", type=int, default=200, help="Bootstrap replicates")
	p.add_argument("--seed", type=int, default=0)
	args = p.parse_args()

	rng = np.random.default_rng(args.seed)
	L_list = [int(x.strip()) for x in args.L.split(",") if x.strip()]
	fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]

	ratios = []
	for _ in range(args.B):
		cfg = ProportionalConfig(lengths=rng.choice(L_list, size=len(L_list), replace=True), fractions=fractions)
		df = proportional_compressibility(cfg)
		# Take mean ratio per replicate
		ratios.append(df["ratio_odd_over_even"].mean())

	ratios_arr = np.array([float(r) for r in ratios])
	mean_r, lo, hi = bootstrap_ratio_ci(ratios_arr)
	out = pl.DataFrame({"mean_ratio": [mean_r], "ci_low": [lo], "ci_high": [hi], "B": [args.B]})
	path = "bootstrap_ratio_ci.csv"
	out.write_csv(path)
	print(f"Saved {path}: mean={mean_r:.3f}, CI=({lo:.3f},{hi:.3f})")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
