"""Thin wrappers around ffmpeg / ffprobe."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 8000


def duration(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def size(path: str | Path) -> tuple[int, int]:
    """Displayed width and height of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:stream_side_data=rotation", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.split()[0].split(",")
    w, h = int(out[0]), int(out[1])
    rotated = len(out) > 2 and out[2].lstrip("-") in {"90", "270"}
    return (h, w) if rotated else (w, h)


def audio_energy(path: str | Path) -> np.ndarray:
    """Loudness (dB RMS) of the audio track, one value per second."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0?", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,  # no audio track -> empty result
    )
    chunk = SAMPLE_RATE * 2  # one second of 16-bit mono
    values = []
    assert proc.stdout is not None
    while True:
        buf = proc.stdout.read(chunk)
        if len(buf) < chunk // 2:  # ignore a trailing fraction of a second
            break
        samples = np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.int16).astype(np.float64)
        rms = np.sqrt(np.mean(samples**2))
        values.append(20 * np.log10(rms + 1.0))
        if len(buf) < chunk:
            break
    proc.wait()
    return np.array(values)


_PTS = re.compile(r"pts_time:([0-9.]+)")


def scene_changes(path: str | Path, n_seconds: int, threshold: float = 0.3) -> np.ndarray:
    """Number of hard cuts / big visual changes in each second."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-an",
         "-vf", f"fps=4,scale=160:-2,select='gt(scene,{threshold})',metadata=print:file=-",
         "-f", "null", "-"],
        capture_output=True, text=True,
    ).stdout
    counts = np.zeros(n_seconds)
    for m in _PTS.finditer(out):
        t = int(float(m.group(1)))
        if t < n_seconds:
            counts[t] += 1
    return counts
