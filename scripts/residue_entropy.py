from __future__ import annotations

import argparse
import polars as pl

from zeta_compress.analysis import residue_class_entropy
from zeta_compress.plotting import plot_entropy_by_residue


def main() -> int:
	p = argparse.ArgumentParser(description="Residue-class spectral entropy for zeta(2) subsequences")
	p.add_argument("--L", type=int, default=4096)
	p.add_argument("--q-list", type=str, default="3,4,6,8", help="Comma-separated q values")
	p.add_argument("--remove-dc", action="store_true")
	p.add_argument("--plot", action="store_true")
	args = p.parse_args()

	q_list = [int(x.strip()) for x in args.q_list.split(",") if x.strip()]
	df = residue_class_entropy(args.L, q_list, remove_dc=bool(args.remove_dc))
	path = f"residue_entropy_L{args.L}.csv"
	df.write_csv(path)
	print(f"Saved {path}")
	if args.plot:
		plot_entropy_by_residue(df, save_path=f"plots/residue_entropy_L{args.L}.png")
		print(f"Saved plots/residue_entropy_L{args.L}.png")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
