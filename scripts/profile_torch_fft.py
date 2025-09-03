from __future__ import annotations

import argparse
import time
import numpy as np

try:
	import torch
	from torch.profiler import profile, ProfilerActivity
	_HAS_TORCH = True
except Exception:
	_HAS_TORCH = False

from zeta_compress import zeta_sequence, even_mask, odd_mask


def run_once(L: int, M: int, device: str) -> float:
	base = zeta_sequence(L, 2.0).astype(np.float64)
	seq = base * even_mask(L)
	if not _HAS_TORCH:
		raise SystemExit("Torch not available")
	dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
	tensor = torch.tensor(seq, dtype=torch.float64, device=dev)
	t0 = time.time()
	with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=True) as prof:
		F = torch.fft.rfft(tensor)
		mag = torch.abs(F)
		order = torch.argsort(mag, descending=True)
		keep = order[:M]
		F_keep = torch.zeros_like(F)
		F_keep[keep] = F[keep]
		recon = torch.fft.irfft(F_keep, n=tensor.shape[0])
	t1 = time.time()
	print(prof.key_averages().table(sort_by="cuda_time_total"))
	return t1 - t0


def main() -> int:
	p = argparse.ArgumentParser(description="Profile torch rFFT top-M reconstruction")
	p.add_argument("--L", type=int, default=1<<16)
	p.add_argument("--M", type=int, default=512)
	p.add_argument("--device", type=str, default=None)
	args = p.parse_args()

	if not _HAS_TORCH:
		print("Torch not available; skipping")
		return 0
	elapsed = run_once(args.L, args.M, args.device)
	print(f"Elapsed {elapsed:.3f}s")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
