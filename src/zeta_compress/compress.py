from __future__ import annotations

from typing import Iterable, Literal, Optional, Tuple
import numpy as np

# Optional SciPy DCT and windows
try:
	try:
		from scipy.fft import dct as _dct, idct as _idct  # type: ignore
	except Exception:  # pragma: no cover
		from scipy.fftpack import dct as _dct, idct as _idct  # type: ignore
	_HAS_DCT = True
except Exception:  # pragma: no cover
	_HAS_DCT = False

try:  # pragma: no cover
	from scipy.signal.windows import tukey as _tukey  # type: ignore
	from scipy.signal.windows import dpss as _dpss  # type: ignore
	_HAS_WIN = True
except Exception:  # pragma: no cover
	_HAS_WIN = False

# Optional torch
try:  # pragma: no cover - optional runtime
	import torch
	_HAS_TORCH = True
except Exception:  # pragma: no cover
	_HAS_TORCH = False

def _parse_window_spec(window: Optional[str]) -> tuple[str, Optional[float]]:
	if window is None:
		return ("none", None)
	spec = window.strip().lower()
	if ":" in spec:
		name, arg = spec.split(":", 1)
		try:
			val = float(arg)
		except Exception:
			val = None
		return (name, val)
	return (spec, None)


def _apply_window(signal: np.ndarray, window: Optional[str]) -> np.ndarray:
	name, arg = _parse_window_spec(window)
	if name in (None, "none"):
		return signal
	if name in ("hann", "hanning"):
		return signal * np.hanning(len(signal))
	if name == "tukey":
		alpha = 0.5 if arg is None else float(arg)
		if _HAS_WIN:
			w = _tukey(len(signal), alpha=alpha)
			return signal * w
		# fallback: approximate with hann if SciPy missing
		return signal * np.hanning(len(signal))
	if name == "dpss":
		NW = 2.5 if arg is None else float(arg)
		if _HAS_WIN:
			# Use the first DPSS taper
			w = _dpss(len(signal), NW=NW, Kmax=1)
			w0 = w[0] if hasattr(w, "__len__") else w
			return signal * np.asarray(w0)
		return signal * np.hanning(len(signal))
	# Unknown window: no-op
	return signal


def preprocess_sequence(sequence: np.ndarray, *, unit_norm: bool = True, remove_dc: bool = False, window: Optional[str] = None) -> np.ndarray:
	seq = np.asarray(sequence)
	if remove_dc:
		seq = seq - np.mean(seq)
	seq = _apply_window(seq, window)
	if unit_norm:
		norm = float(np.linalg.norm(seq))
		if norm > 0.0:
			seq = seq / norm
	return seq


Basis = Literal["rfft", "dct"]
Backend = Literal["numpy", "torch"]


def _transform_numpy(signal: np.ndarray, basis: Basis) -> Tuple[np.ndarray, int]:
	if basis == "rfft":
		return np.fft.rfft(signal), len(signal)
	if basis == "dct":
		if not _HAS_DCT:
			return np.fft.rfft(signal), len(signal)
		return _dct(signal, type=2, norm="ortho"), len(signal)
	return np.fft.rfft(signal), len(signal)


def _inverse_top_m_numpy(transform_coeffs: np.ndarray, keep_indices: np.ndarray, basis: Basis, n: int) -> np.ndarray:
	if basis == "rfft" or (basis == "dct" and not _HAS_DCT):
		F = np.zeros_like(transform_coeffs)
		F[keep_indices] = transform_coeffs[keep_indices]
		return np.fft.irfft(F, n=n)
	# DCT-II inverse via IDCT-II
	coeffs = np.zeros_like(transform_coeffs)
	coeffs[keep_indices] = transform_coeffs[keep_indices]
	return _idct(coeffs, type=2, norm="ortho")


def top_m_rel_error(
	seq: np.ndarray,
	M: int,
	*,
	basis: Basis = "rfft",
	backend: Backend = "numpy",
	device: Optional[str] = None,
	unit_norm: bool = True,
	remove_dc: bool = False,
	window: Optional[str] = None,
) -> float:
	"""Relative L2 error after keeping top-M coefficients by magnitude.

	If backend="torch" and torch is available, uses torch.fft on given device.
	"""
	prepared = preprocess_sequence(seq, unit_norm=unit_norm, remove_dc=remove_dc, window=window)
	if backend == "torch":  # pragma: no cover - optional
		assert _HAS_TORCH, "PyTorch not available"
		dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
		tensor = torch.as_tensor(prepared, device=dev)
		if basis == "rfft":
			F = torch.fft.rfft(tensor)
		else:
			# No torch DCT in standard API; fall back to numpy for DCT
			F_np, n = _transform_numpy(prepared, basis)
			order = np.argsort(np.abs(F_np).astype(np.float64))[::-1]
			keep = order[: max(1, int(M))]
			recon = _inverse_top_m_numpy(F_np, keep, basis=basis, n=n)
			num = float(np.linalg.norm(prepared - recon))
			den = float(np.linalg.norm(prepared))
			return (num / den) if den != 0.0 else 0.0
		# rfft path
		magnitudes = torch.abs(F)
		order = torch.argsort(magnitudes, descending=True)
		keep = order[: max(1, int(M))]
		F_keep = torch.zeros_like(F)
		F_keep[keep] = F[keep]
		recon = torch.fft.irfft(F_keep, n=tensor.shape[0])
		num = torch.linalg.vector_norm(tensor - recon).item()
		den = torch.linalg.vector_norm(tensor).item()
		return (num / den) if den != 0.0 else 0.0
	# numpy path
	T, n = _transform_numpy(prepared, basis)
	order = np.argsort(np.abs(T))[::-1]
	keep = order[: max(1, int(M))]
	recon = _inverse_top_m_numpy(T, keep, basis=basis, n=n)
	num = float(np.linalg.norm(prepared - recon))
	den = float(np.linalg.norm(prepared))
	return (num / den) if den != 0.0 else 0.0


def spectral_entropy(seq: np.ndarray, *, base: float = 2.0) -> float:
	"""Shannon entropy of normalized power spectrum (rFFT)."""
	prepared = preprocess_sequence(seq, unit_norm=True, remove_dc=False, window=None)
	F = np.fft.rfft(prepared)
	power = np.abs(F) ** 2
	total = float(power.sum())
	if total == 0.0:
		return 0.0
	p = power / total
	p = np.where(p > 0, p, 1.0)
	H = float(-(p * (np.log(p) / np.log(base))).sum())
	return H


def min_modes_for_error(seq: np.ndarray, eps: float) -> int:
	"""Minimum number of modes (energy criterion) to achieve relative L2 error ≤ eps."""
	assert eps >= 0.0
	prepared = preprocess_sequence(seq, unit_norm=True, remove_dc=False, window=None)
	F = np.fft.rfft(prepared)
	energy = np.abs(F) ** 2
	order = np.argsort(energy)[::-1]
	sorted_energy = energy[order]
	cum = np.cumsum(sorted_energy)
	total = float(energy.sum())
	if total == 0.0:
		return 0
	target = (1.0 - eps * eps) * total
	idx = int(np.searchsorted(cum, target, side="left"))
	M = int(min(max(idx + 1, 0), len(sorted_energy)))
	return M
