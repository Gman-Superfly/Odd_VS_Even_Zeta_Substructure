from __future__ import annotations

import argparse
import numpy as np

import even_odd_zeta_fft as m


def parse_comma_ints(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_comma_floats(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [float(p) for p in parts]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute minimum Fourier modes M to reach target relative error eps for even vs odd subsequences."
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="1024,2048,4096",
        help="Comma-separated list of lengths L (default: 1024,2048,4096)",
    )
    parser.add_argument(
        "--eps",
        type=str,
        default="1e-1,3e-2",
        help="Comma-separated list of epsilon targets (default: 1e-1,3e-2)",
    )
    parser.add_argument(
        "--remove-dc",
        action="store_true",
        help="Remove the mean before spectral analysis (DC removed)",
    )
    args = parser.parse_args()

    L_list = parse_comma_ints(args.lengths)
    eps_list = parse_comma_floats(args.eps)
    remove_dc = bool(args.remove_dc)

    print("Min modes M for target relative error eps (energy criterion)")
    print(f"Settings: remove_dc={remove_dc}")
    for L in L_list:
        j = np.arange(L, dtype=np.float64)
        base = 1.0 / ((j + 1.0) ** 2)
        even_mask = ((j + 1) % 2 == 0).astype(np.float64)
        odd_mask = 1.0 - even_mask
        seq_even = base * even_mask
        seq_odd = base * odd_mask
        if remove_dc:
            seq_even = seq_even - float(np.mean(seq_even))
            seq_odd = seq_odd - float(np.mean(seq_odd))
        line = [f"L={L:4d}"]
        for eps in eps_list:
            M_even = m.min_modes_for_error(seq_even, eps)
            M_odd = m.min_modes_for_error(seq_odd, eps)
            ratio = (M_odd / M_even) if M_even > 0 else float("inf")
            line.append(
                f"eps={eps:.0e}: M_even={M_even:4d} M_odd={M_odd:4d} ratio={ratio:.2f}"
            )
        print("  ".join(line))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


