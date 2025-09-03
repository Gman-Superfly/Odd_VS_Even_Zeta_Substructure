from __future__ import annotations

import numpy as np
from zeta_compress import even_mask, odd_mask, residue_mask, exact_zeta2_even, exact_zeta2_odd


def test_even_odd_mask_basic() -> None:
	L = 10
	e = even_mask(L)
	o = odd_mask(L)
	assert e.shape == (L,)
	assert o.shape == (L,)
	# Disjoint and cover
	assert np.all((e + o) == 1.0)
	# First index is odd (1), second is even (2)
	assert o[0] == 1.0 and e[0] == 0.0
	assert e[1] == 1.0 and o[1] == 0.0


def test_residue_mask() -> None:
	L = 12
	q = 3
	for r in range(q):
		m = residue_mask(L, q, r)
		assert m.shape == (L,)
		idx = np.where(m == 1.0)[0]
		# n runs 1..L, adjust index to n
		ns = idx + 1
		assert np.all((ns % q) == r)


def test_exact_zeta2_splits() -> None:
	# pi^2/24 and pi^2/8 consistency and ratio 3 between odd and even sums
	e = exact_zeta2_even()
	o = exact_zeta2_odd()
	assert e > 0 and o > 0
	r = o / e
	assert np.isclose(r, 3.0, rtol=1e-12, atol=0.0)
