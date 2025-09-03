from __future__ import annotations

import argparse
import time
import numpy as np

try:
	import torch
	_HAS_TORCH = True
except Exception:
	_HAS_TORCH = False

from zeta_compress import zeta_sequence, even_mask, odd_mask
from zeta_compress import top_m_rel_error


def main() -> int:
	p = argparse.ArgumentParser(description="Toy ML example using zeta encodings and torch rFFT")
	p.add_argument("--L", type=int, default=16384)
	p.add_argument("--M", type=int, default=128)
	p.add_argument("--device", type=str, default=None)
	args = p.parse_args()

	L = args.L
	M = args.M
	base = zeta_sequence(L, 2.0).astype(np.float64)
	seq_even = base * even_mask(L)
	seq_odd = base * odd_mask(L)

	if _HAS_TORCH:
		dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
		t0 = time.time()
		err_e = top_m_rel_error(seq_even, M, basis="rfft", backend="torch", device=str(dev), unit_norm=True)
		err_o = top_m_rel_error(seq_odd, M, basis="rfft", backend="torch", device=str(dev), unit_norm=True)
		t1 = time.time()
		print(f"torch[{dev}] M={M} even_err={err_e:.3e} odd_err={err_o:.3e} ratio={err_o/err_e:.2f} time={t1-t0:.3f}s")
	else:
		t0 = time.time()
		err_e = top_m_rel_error(seq_even, M, basis="rfft", backend="numpy", unit_norm=True)
		err_o = top_m_rel_error(seq_odd, M, basis="rfft", backend="numpy", unit_norm=True)
		t1 = time.time()
		print(f"numpy M={M} even_err={err_e:.3e} odd_err={err_o:.3e} ratio={err_o/err_e:.2f} time={t1-t0:.3f}s")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
