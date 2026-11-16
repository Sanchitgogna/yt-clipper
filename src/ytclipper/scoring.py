"""Turn raw per-second signals into a highlight score and pick clip windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# How much each signal counts. Audience signals (what viewers replayed, when chat
# exploded) are better evidence of a "good scene" than raw loudness or cuts.
DEFAULT_WEIGHTS = {"heatmap": 1.5, "chat": 1.5, "audio": 1.0, "scenes": 0.5}
# Smallest spread treated as meaningful, so a very steady signal does not turn
# its own tiny wobble into "peaks" (audio is in dB).
MIN_SCALE = {"audio": 1.5}


@dataclass
class Clip:
    start: float
    end: float
    peak: float   # second of the highest score inside the clip
    score: float

    @property
    def length(self) -> float:
        return self.end - self.start


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) == 0:
        return x.astype(float)
    kernel = np.ones(window) / window
    padded = np.pad(x.astype(float), (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def robust_z(x: np.ndarray, min_scale: float = 0.0, baseline: np.ndarray | None = None) -> np.ndarray:
    """How far each value sits above the typical level (median/MAD based)."""
    ref = x if baseline is None else baseline
    if len(ref) == 0:
        return np.zeros(len(x))
    med = np.median(ref)
    scale = 1.4826 * np.median(np.abs(ref - med))
    if scale < 1e-9:
        scale = np.std(ref)
    scale = max(scale, min_scale)
    if scale < 1e-9:
        return np.zeros(len(x))
    return (x - med) / scale


def fit(x: np.ndarray, n: int) -> np.ndarray:
    """Pad or trim a signal to exactly n seconds."""
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)), mode="edge" if len(x) else "constant")


def combine(signals: dict[str, np.ndarray], weights: dict[str, float] | None = None,
            window: int = 5) -> np.ndarray:
    """Weighted mean of the normalised, smoothed signals."""
    weights = weights or DEFAULT_WEIGHTS
    n = max(len(s) for s in signals.values())
    total, wsum = np.zeros(n), 0.0
    for name, raw in signals.items():
        w = weights.get(name, 1.0)
        if w <= 0:
            continue
        z = robust_z(smooth(fit(raw, n), window), MIN_SCALE.get(name, 0.0))
        total += w * np.clip(z, 0, 8)
        wsum += w
    return total / wsum if wsum else total


def pick_clips(
    score: np.ndarray,
    count: int = 5,
    min_len: float = 15,
    max_len: float = 60,
    lead_in: float = 4,
    tail: float = 2,
    gap: float = 10,
    min_score: float = 0.5,
) -> list[Clip]:
    """Greedily take the highest peaks and grow a window around each one."""
    n = len(score)
    remaining = score.astype(float).copy()
    clips: list[Clip] = []
    while len(clips) < count and n:
        p = int(np.argmax(remaining))
        if remaining[p] < min_score:
            break
        thr = score[p] * 0.5
        a = b = p
        while a > 0 and score[a - 1] >= thr and remaining[a - 1] > -np.inf:
            a -= 1
        while b < n - 1 and score[b + 1] >= thr and remaining[b + 1] > -np.inf:
            b += 1
        start, end = a - lead_in, b + 1 + tail
        if end - start < min_len:  # the build-up matters more than the aftermath
            short = min_len - (end - start)
            start, end = start - 0.6 * short, end + 0.4 * short
        if end - start > max_len:  # keep the peak, biased towards the lead-up
            start = max(start, p - 0.6 * max_len)
            end = start + max_len
        if start < 0:
            start, end = 0.0, min(float(n), end - start)
        if end > n:
            start, end = max(0.0, start - (end - n)), float(n)
        clips.append(Clip(float(start), float(end), float(p), float(score[p])))
        lo, hi = max(0, int(start - gap)), min(n, int(np.ceil(end + gap)))
        remaining[lo:hi] = -np.inf
    return sorted(clips, key=lambda c: c.start)
