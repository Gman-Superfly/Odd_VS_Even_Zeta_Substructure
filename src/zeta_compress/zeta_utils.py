from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
from mpmath import mp, zeta as mp_zeta

ComplexLike = Union[float, complex]

# High precision for reference checks
mp.dps = 80

__all__ = [
	"even_mask",
	"odd_mask",
	"residue_mask",
	"zeta_sequence",
	"zeta_sequence_masked",
	"exact_zeta2_even",
	"exact_zeta2_odd",
]


def even_mask(length: int) -> np.ndarray:
	"""Return a float mask of length L selecting even indices n=2,4,6,...

	Args:
		length: Sequence length L (n runs from 1..L)
	Returns:
		Mask array of shape (L,), values in {0.0, 1.0}
	Raises:
		AssertionError: if length < 1
	"""
	assert isinstance(length, int) and length >= 1, "length must be positive int"
	j = np.arange(1, length + 1, dtype=np.int64)
	return (j % 2 == 0).astype(np.float64)


def odd_mask(length: int) -> np.ndarray:
	"""Return a float mask selecting odd indices n=1,3,5,..."""
	assert isinstance(length, int) and length >= 1
	j = np.arange(1, length + 1, dtype=np.int64)
	return (j % 2 == 1).astype(np.float64)


def residue_mask(length: int, modulus: int, residue: int) -> np.ndarray:
	"""Return mask for n ≡ residue (mod modulus).

	Args:
		length: number of terms L
		modulus: q ≥ 2
		residue: 0 ≤ r < q
	"""
	assert isinstance(length, int) and length >= 1
	assert isinstance(modulus, int) and modulus >= 2
	assert isinstance(residue, int) and 0 <= residue < modulus
	j = np.arange(1, length + 1, dtype=np.int64)
	return ((j % modulus) == residue).astype(np.float64)


def zeta_sequence(length: int, s: ComplexLike) -> np.ndarray:
	"""Return the length-L sequence a_n = n^{-s} for n=1..L.

	Supports real s>1 and complex s, e.g., s = 0.5 + 14.1347j.
	Values are computed in complex128 when s is complex, else float64.
	"""
	assert isinstance(length, int) and length >= 1
	n = np.arange(1, length + 1)
	if isinstance(s, complex):
		arr = n.astype(np.complex128)
		return arr ** (-s)
	# float-like
	arr = n.astype(np.float64)
	return arr ** (float(-s))


def zeta_sequence_masked(length: int, s: ComplexLike, mask: Optional[np.ndarray]) -> np.ndarray:
	"""Return masked Dirichlet series terms: mask * n^{-s}.

	If mask is None, returns n^{-s}.
	"""
	seq = zeta_sequence(length, s)
	if mask is None:
		return seq
	m = np.asarray(mask)
	assert m.shape == (length,), "mask shape mismatch"
	# Promote to complex if needed
	if np.iscomplexobj(seq) or np.iscomplexobj(m):
		return np.asarray(m, dtype=np.complex128) * np.asarray(seq, dtype=np.complex128)
	return np.asarray(m, dtype=np.float64) * np.asarray(seq, dtype=np.float64)


def exact_zeta2_even() -> float:
	"""Exact value of sum_{k>=1} (2k)^{-2} = (1/4) zeta(2) = pi^2/24."""
	return float((mp.pi ** 2) / 24)


def exact_zeta2_odd() -> float:
	"""Exact value of sum_{k>=1} (2k-1)^{-2} = (1 - 2^{-2}) zeta(2) = pi^2/8."""
	return float((mp.pi ** 2) / 8)
