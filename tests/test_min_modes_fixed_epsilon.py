from __future__ import annotations

import numpy as np

from zeta_compress import (
	even_mask,
	odd_mask,
	zeta_sequence,
	min_modes_for_error,
)


def test_min_modes_fixed_epsilon_parity_comparison() -> None:
	L_list = [2048, 4096, 8192]
	eps_list = [1e-1, 3e-2]
	for L in L_list:
		base = zeta_sequence(L, 2.0)
		seq_even = base * even_mask(L).astype(base.dtype)
		seq_odd = base * odd_mask(L).astype(base.dtype)
		# Bounds and ordering at fixed eps
		for eps in eps_list:
			M_even = min_modes_for_error(seq_even, eps)
			M_odd = min_modes_for_error(seq_odd, eps)
			n_bins = L // 2 + 1
			assert 0 <= M_even <= n_bins
			assert 0 <= M_odd <= n_bins
			# Parity effect: odd typically needs at least as many modes as even
			assert M_odd >= M_even
		# Monotonicity in eps (harder target -> more modes)
		M_even_lo = min_modes_for_error(seq_even, eps_list[0])
		M_even_hi = min_modes_for_error(seq_even, eps_list[1])
		M_odd_lo = min_modes_for_error(seq_odd, eps_list[0])
		M_odd_hi = min_modes_for_error(seq_odd, eps_list[1])
		# Note: eps_list[1] < eps_list[0]
		assert M_even_hi >= M_even_lo
		assert M_odd_hi >= M_odd_lo


