from __future__ import annotations

import numpy as np
from zeta_compress import top_m_rel_error, min_modes_for_error, preprocess_sequence


def test_top_m_error_monotone() -> None:
	L = 1024
	j = np.arange(L, dtype=np.float64)
	base = 1.0 / ((j + 1.0) ** 2)
	errs = []
	for M in (4, 8, 16, 32, 64):
		errs.append(top_m_rel_error(base, M, basis="rfft", backend="numpy", unit_norm=True))
	# Nonincreasing
	for a, b in zip(errs, errs[1:]):
		assert b <= a + 1e-12


def test_min_modes_for_error_bounds() -> None:
	L = 2048
	seq = preprocess_sequence(1.0 / (np.arange(L, dtype=np.float64) + 1.0) ** 2, unit_norm=True, remove_dc=False)
	m1 = min_modes_for_error(seq, 0.5)
	m2 = min_modes_for_error(seq, 0.2)
	assert 0 <= m1 <= m2 <= (L // 2 + 1)
