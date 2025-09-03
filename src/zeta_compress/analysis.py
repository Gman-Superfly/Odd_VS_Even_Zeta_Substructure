from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union
from datetime import datetime
import numpy as np
import polars as pl

from .compress import top_m_rel_error, spectral_entropy, preprocess_sequence
from .zeta_utils import even_mask, odd_mask, residue_mask, zeta_sequence


@dataclass
class ProportionalConfig:
	lengths: Sequence[int]
	fractions: Sequence[float]
	s: Union[float, complex] = 2.0
	basis: str = "rfft"
	backend: str = "numpy"
	remove_dc: bool = False
	window: Optional[str] = None


def proportional_compressibility(cfg: ProportionalConfig) -> pl.DataFrame:
	rows: List[dict] = []
	for L in cfg.lengths:
		base = zeta_sequence(L, cfg.s)
		if np.iscomplexobj(base):
			base = base.real
		seq_even = base * even_mask(L).astype(base.dtype)
		seq_odd = base * odd_mask(L).astype(base.dtype)
		n_bins = L // 2 + 1
		for frac in cfg.fractions:
			M = max(1, int(frac * n_bins))
			err_e = top_m_rel_error(
				seq_even, M, basis=cfg.basis, backend=cfg.backend, unit_norm=True, remove_dc=cfg.remove_dc, window=cfg.window
			)
			err_o = top_m_rel_error(
				seq_odd, M, basis=cfg.basis, backend=cfg.backend, unit_norm=True, remove_dc=cfg.remove_dc, window=cfg.window
			)
			row = {
				"L": int(L),
				"fraction": float(frac),
				"M": int(M),
				"even_rel_err": float(err_e),
				"odd_rel_err": float(err_o),
				"ratio_odd_over_even": float((err_o / err_e) if err_e > 0 else np.inf),
			}
			# Store s as real/imag components for other consumers; will be dropped for legacy CSV
			if isinstance(cfg.s, complex):
				row["s_real"] = float(cfg.s.real)
				row["s_imag"] = float(cfg.s.imag)
			else:
				row["s_real"] = float(cfg.s)
				row["s_imag"] = 0.0
			rows.append(row)
	return pl.DataFrame(rows)


def save_dataframe_with_timestamp(df: pl.DataFrame, stem: str) -> str:
	# Match legacy schema for proportional results: drop s_* columns when saving that stem
	if stem.startswith("results_compressibility_proportional"):
		keep_cols = ["L", "fraction", "M", "even_rel_err", "odd_rel_err", "ratio_odd_over_even"]
		df = df.select([c for c in keep_cols if c in df.columns])
		# Ensure numeric types
		df = df.with_columns([
			pl.col("L").cast(pl.Int64),
			pl.col("M").cast(pl.Int64),
			pl.col("fraction").cast(pl.Float64),
			pl.col("even_rel_err").cast(pl.Float64),
			pl.col("odd_rel_err").cast(pl.Float64),
			pl.col("ratio_odd_over_even").cast(pl.Float64),
		])
		# Format to legacy strings via map_elements
		df = df.with_columns([
			pl.col("fraction").map_elements(lambda x: f"{x:.3f}", return_dtype=pl.Utf8),
			pl.col("even_rel_err").map_elements(lambda x: f"{x:.6e}", return_dtype=pl.Utf8),
			pl.col("odd_rel_err").map_elements(lambda x: f"{x:.6e}", return_dtype=pl.Utf8),
			pl.col("ratio_odd_over_even").map_elements(lambda x: f"{x:.3f}", return_dtype=pl.Utf8),
		])
	# Ensure no object dtypes that CSV writer dislikes (cast such cols to Utf8)
	object_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Object]
	if object_cols:
		df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in object_cols])
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	path = f"{stem}_{stamp}.csv"
	df.write_csv(path)
	return path


def residue_class_entropy(length: int, q_list: Sequence[int], remove_dc: bool = False) -> pl.DataFrame:
	rows: List[dict] = []
	base = zeta_sequence(length, 2.0)
	for q in q_list:
		for r in range(q):
			mask = residue_mask(length, q, r).astype(base.dtype)
			seq = preprocess_sequence(base * mask, unit_norm=True, remove_dc=remove_dc, window=None)
			H = spectral_entropy(seq, base=2.0)
			rows.append({"L": int(length), "q": int(q), "r": int(r), "entropy": float(H), "effective_modes": float(2.0 ** H)})
	return pl.DataFrame(rows)
