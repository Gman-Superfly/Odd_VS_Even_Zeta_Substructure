from __future__ import annotations

import argparse
from typing import List, Union
import numpy as np
import polars as pl

from zeta_compress.analysis import ProportionalConfig, proportional_compressibility, save_dataframe_with_timestamp
from zeta_compress.plotting import plot_error_curves


def parse_csv_ints(text: str) -> List[int]:
	return [int(p.strip()) for p in text.split(",") if p.strip()]


def parse_csv_floats(text: str) -> List[float]:
	return [float(p.strip()) for p in text.split(",") if p.strip()]


def parse_s(text: str) -> Union[float, complex]:
	# Accept float like "2" or complex like "0.5+14.134j"
	try:
		return float(text)
	except ValueError:
		return complex(text)


def main() -> int:
	p = argparse.ArgumentParser(description="Zeta subsequences compressibility experiments")
	p.add_argument("--lengths", type=str, default="4096,16384,65536,262144,524288", help="Comma-separated L values")
	p.add_argument("--fractions", type=str, default="0.001,0.002,0.005,0.01,0.02", help="Comma-separated fractions of rFFT bins")
	p.add_argument("--s", type=str, default="2.0", help="Exponent s (real e.g. 3, or complex e.g. 0.5+14.134j). Complex uses real part for spectrum.")
	p.add_argument("--basis", type=str, default="rfft", choices=["rfft", "dct"], help="Transform basis")
	p.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"], help="Computation backend")
	p.add_argument("--remove-dc", action="store_true", help="Remove mean before analysis")
	p.add_argument("--window", type=str, default=None, help="Optional window name (hann)")
	p.add_argument("--plot", action="store_true", help="Generate plots to ./plots")
	args = p.parse_args()

	s_value = parse_s(args.s)
	cfg = ProportionalConfig(
		lengths=parse_csv_ints(args.lengths),
		fractions=parse_csv_floats(args.fractions),
		s=s_value,
		basis=args.basis,
		backend=args.backend,
		remove_dc=bool(args.remove_dc),
		window=args.window,
	)
	df = proportional_compressibility(cfg)
	out = save_dataframe_with_timestamp(df, "results_compressibility_proportional")
	print(f"Saved results to {out}")
	if args.plot:
		plot_error_curves(df.with_columns(pl.col("M").cast(pl.Int64)), title=f"Even vs Odd compressibility (s={args.s})", save_path="plots/err_curves.png")
		print("Saved plot to plots/err_curves.png")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
