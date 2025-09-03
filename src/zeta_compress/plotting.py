from __future__ import annotations

from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

sns.set_context("talk")

__all__ = ["plot_error_curves", "plot_entropy_by_residue", "plot_ratio_vs_t"]


def _unique_sorted(series: pl.Series) -> List:
	vals = series.unique().to_list()
	try:
		return sorted(vals)
	except Exception:
		return vals


def plot_error_curves(df: pl.DataFrame, *, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
	"""Plot relative L2 error vs M for even/odd from a Polars DataFrame.

	Expected columns: L (int), M (int), even_rel_err (float), odd_rel_err (float).
	"""
	Ls = _unique_sorted(df.get_column("L"))
	fig, axes = plt.subplots(len(Ls), 1, figsize=(8, 4 * max(1, len(Ls))), sharex=True)
	if not isinstance(axes, (list, np.ndarray)):
		axes = [axes]
	for ax, L in zip(axes, Ls):
		d = df.filter(pl.col("L") == L).select(["M", "even_rel_err", "odd_rel_err"]).sort("M")
		M = d.get_column("M").to_numpy()
		even = d.get_column("even_rel_err").to_numpy()
		odd = d.get_column("odd_rel_err").to_numpy()
		ax.plot(M, even, label="Even", marker="o")
		ax.plot(M, odd, label="Odd", marker="s")
		ax.set_ylabel("Rel L2 error")
		ax.set_title(f"L={L}")
	axes[-1].set_xlabel("M (kept modes)")
	axes[0].legend()
	if title:
		fig.suptitle(title)
	fig.tight_layout()
	if save_path:
		fig.savefig(save_path, dpi=160)


def plot_entropy_by_residue(df: pl.DataFrame, *, save_path: Optional[str] = None) -> None:
	"""Bar plot of entropy per residue class grouped by q from Polars DataFrame.

	Expected columns: q (int), r (int), entropy (float).
	"""
	Qs = _unique_sorted(df.get_column("q"))
	num_cols = len(Qs)
	fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols + 2, 4), squeeze=False)
	axes = axes[0]
	for ax, q in zip(axes, Qs):
		d = df.filter(pl.col("q") == q).select(["r", "entropy"]).sort("r")
		r_vals = d.get_column("r").to_numpy()
		H = d.get_column("entropy").to_numpy()
		ax.bar(r_vals, H, width=0.8)
		ax.set_title(f"q={q}")
		ax.set_xlabel("residue r")
		ax.set_ylabel("entropy (bits)")
	fig.tight_layout()
	if save_path:
		fig.savefig(save_path, dpi=160)


def plot_ratio_vs_t(df: pl.DataFrame, *, save_path: Optional[str] = None, title: Optional[str] = None) -> None:
	"""Plot ratio_odd_over_even vs t, lines per fraction, averaged when multiple rows per (t, fraction).

	Expected columns: t (float), fraction (float), ratio_odd_over_even (float).
	"""
	agg = df.group_by(["t", "fraction"]).agg(pl.col("ratio_odd_over_even").mean().alias("ratio"))
	fractions = _unique_sorted(agg.get_column("fraction"))
	plt.figure(figsize=(9, 5))
	for frac in fractions:
		d = agg.filter(pl.col("fraction") == frac).select(["t", "ratio"]).sort("t")
		t_vals = d.get_column("t").to_numpy()
		r_vals = d.get_column("ratio").to_numpy()
		plt.plot(t_vals, r_vals, label=f"frac={frac}")
	plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
	plt.xlabel("t in s=1/2+it")
	plt.ylabel("mean ratio (odd/even)")
	if title:
		plt.title(title)
	plt.legend()
	plt.tight_layout()
	if save_path:
		plt.savefig(save_path, dpi=160)
