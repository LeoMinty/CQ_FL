"""Prepare the public DroneRF L/H-band CSV files for CQ-FL experiment one.

The public files contain two real-valued RF bands (L and H), not raw I/Q pairs.
For the draft's complex CNN, this script keeps the complex FFT coefficients of
both bands: 1024 bins from L followed by 1024 bins from H, reshaped to 64x32.
Every derived window retains its physical-segment id so train/test and client
splits cannot leak windows from the same recording.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


FILE_RE = re.compile(r"(?P<bui>[01]{5})(?P<band>[LH])[_-]?(?P<segment>\d+)$", re.IGNORECASE)


def drf2_label(bui: str) -> int:
    value = int(bui, 2)
    if value == 0:
        return 0  # no drone
    if 16 <= value <= 19:
        return 1  # Bebop
    if 20 <= value <= 23:
        return 2  # AR Drone
    if value == 24:
        return 3  # Phantom
    raise ValueError(f"unsupported DroneRF BUI code: {bui}")


def discover_pairs(root: Path) -> Dict[Tuple[str, str], Dict[str, Path]]:
    pairs: Dict[Tuple[str, str], Dict[str, Path]] = {}
    for path in root.rglob("*.csv"):
        match = FILE_RE.search(path.stem)
        if not match:
            continue
        key = (match.group("bui"), match.group("segment"))
        pairs.setdefault(key, {})[match.group("band").upper()] = path
    return {key: value for key, value in pairs.items() if {"L", "H"}.issubset(value)}


def read_signal(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return np.asarray(values, dtype=np.float32).reshape(-1)


def complex_spectrum(window: np.ndarray, bins: int) -> np.ndarray:
    centered = window - np.mean(window, dtype=np.float64)
    # The source is real-valued, so rfft avoids computing the conjugate half.
    spectrum = np.fft.rfft(centered)
    if spectrum.size < bins:
        spectrum = np.pad(spectrum, (0, bins - spectrum.size))
    return spectrum[:bins].astype(np.complex64)


def prepare(root: Path, output: Path, window_size: int, bins_per_band: int, max_windows: int):
    pairs = discover_pairs(root)
    if not pairs:
        raise FileNotFoundError(f"no paired DroneRF L/H CSV files found below {root}")
    features, labels, groups = [], [], []
    for (bui, segment), bands in sorted(pairs.items()):
        low = read_signal(bands["L"])
        high = read_signal(bands["H"])
        usable = min(low.size, high.size) // window_size
        if max_windows:
            usable = min(usable, max_windows)
        for window_id in range(usable):
            start = window_id * window_size
            end = start + window_size
            low_fft = complex_spectrum(low[start:end], bins_per_band)
            high_fft = complex_spectrum(high[start:end], bins_per_band)
            spectrum = np.concatenate([low_fft, high_fft])
            if spectrum.size != 2048:
                raise ValueError("the draft model adapter expects 2 x 1024 = 2048 bins")
            pair = np.stack([spectrum.real, spectrum.imag], axis=-1)
            features.append(pair.reshape(64, 32, 1, 2))
            labels.append(drf2_label(bui))
            groups.append(f"{bui}_{segment}")
        print(f"{bui}_{segment}: {usable} windows")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups, dtype="U32")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=x,
        Y=y,
        groups=group_array,
        class_names=np.asarray(["Background", "Bebop", "AR", "Phantom"]),
        representation="complex_fft_L1024_H1024",
        window_size=np.int64(window_size),
    )
    print(f"saved {len(y)} windows to {output}")
    print(f"class counts: {np.bincount(y, minlength=4).tolist()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dronerf_processed/dronerf_drf2_complex.npz"),
    )
    parser.add_argument("--window-size", type=int, default=100_000)
    parser.add_argument("--bins-per-band", type=int, default=1024, choices=[1024])
    parser.add_argument("--max-windows-per-segment", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(
        args.raw_root,
        args.output,
        args.window_size,
        args.bins_per_band,
        args.max_windows_per_segment,
    )
