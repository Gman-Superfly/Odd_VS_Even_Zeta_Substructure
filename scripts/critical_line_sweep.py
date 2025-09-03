from __future__ import annotations

import argparse
import numpy as np
import polars as pl

from zeta_compress.analysis import ProportionalConfig, proportional_compressibility, save_dataframe_with_timestamp
from zeta_compress.plotting import plot_ratio_vs_t


def main() -> int:
	p = argparse.ArgumentParser(description="Sweep s=1/2+it for proportional compressibility and plot ratio vs t")
	p.add_argument("--lengths", type=str, default="4096,8192,16384")
	p.add_argument("--fractions", type=str, default="0.002,0.005")
	p.add_argument("--t-min", type=float, default=0.0)
	p.add_argument("--t-max", type=float, default=50.0)
	p.add_argument("--t-steps", type=int, default=26)
	p.add_argument("--avg-windows", type=int, default=1, help="Average results over this many random windows")
	p.add_argument("--window", type=str, default=None, help="Window spec: hann | tukey:alpha | dpss:NW")
	p.add_argument("--remove-dc", action="store_true")
	p.add_argument("--plot", action="store_true")
	p.add_argument("--seed", type=int, default=0)
	args = p.parse_args()

	rng = np.random.default_rng(args.seed)
	L_list = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
	fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
	t_grid = np.linspace(args.t_min, args.t_max, args.t_steps)

	frames: list[pl.DataFrame] = []
	for t in t_grid:
		agg_frames: list[pl.DataFrame] = []
		for _ in range(max(1, int(args.avg_windows))):
			# For averaging, we jitter lengths by small random reductions to simulate distinct windows
			Ls = [max(32, L - int(rng.integers(0, min(128, L//16)))) for L in L_list]
			s = 0.5 + 1j * float(t)
			cfg = ProportionalConfig(lengths=Ls, fractions=fractions, s=s, remove_dc=bool(args.remove_dc), window=args.window)
			df = proportional_compressibility(cfg)
			df = df.with_columns(pl.lit(float(t)).alias("t"))
			agg_frames.append(df)
		frames.append(pl.concat(agg_frames, how="vertical_relaxed"))

	out_df = pl.concat(frames, how="vertical_relaxed")
	path = save_dataframe_with_timestamp(out_df, "critical_line_ratio")
	print(f"Saved {path}")
	if args.plot:
		plot_ratio_vs_t(out_df, save_path="plots/ratio_vs_t.png", title="Odd/Even ratio vs t on s=1/2+it")
		print("Saved plots/ratio_vs_t.png")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
