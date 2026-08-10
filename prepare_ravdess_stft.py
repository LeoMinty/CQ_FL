"""Prepare the 8-class RAVDESS complex STFT input specified in the draft."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np


def fit_shape(spectrum: np.ndarray, frequency_bins: int, frames: int) -> np.ndarray:
    spectrum = spectrum[:frequency_bins, :frames]
    pad_f = max(0, frequency_bins - spectrum.shape[0])
    pad_t = max(0, frames - spectrum.shape[1])
    return np.pad(spectrum, ((0, pad_f), (0, pad_t)))


def prepare(root: Path, output: Path, frequency_bins: int, frames: int):
    features, labels, actors = [], [], []
    for actor_dir in sorted(root.glob("Actor_*")):
        actor = int(actor_dir.name.split("_")[-1])
        for wav in sorted(actor_dir.glob("*.wav")):
            parts = wav.stem.split("-")
            if len(parts) != 7:
                continue
            emotion = int(parts[2]) - 1
            audio, sample_rate = librosa.load(wav, sr=16_000, duration=3.0)
            stft = librosa.stft(audio, n_fft=512, hop_length=256, win_length=512)
            stft = fit_shape(stft, frequency_bins, frames)
            feature = np.stack([stft.real, stft.imag], axis=-1)[..., None, :]
            features.append(feature)
            labels.append(emotion)
            actors.append(actor)
    if not features:
        raise FileNotFoundError(f"no RAVDESS wav files found below {root}")
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    actor_ids = np.asarray(actors, dtype=np.int64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, Xc=x, Y=y, actor=actor_ids, n_classes=np.int64(8))
    print(f"saved {len(y)} samples to {output}")
    print(f"shape={x.shape}, class counts={np.bincount(y, minlength=8).tolist()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ravdess_processed/ravdess_c3_stft.npz"),
    )
    parser.add_argument("--frequency-bins", type=int, default=128)
    parser.add_argument("--frames", type=int, default=192)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.raw_root, args.output, args.frequency_bins, args.frames)
