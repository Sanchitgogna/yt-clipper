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


def audio_energy(path: str | Path) -> np.ndarray:
    """Loudness (dB RMS) of the audio track, one value per second."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        stdout=subprocess.PIPE,
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


def cut(
    src: str | Path,
    dst: str | Path,
    start: float,
    length: float,
    vertical: bool = False,
    concat_list: bool = False,
) -> Path:
    """Cut [start, start+length) out of src, re-encoding for frame-accurate edges."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if concat_list:
        cmd += ["-f", "concat", "-safe", "0"]
    cmd += ["-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}"]
    if vertical:  # centre-crop to 9:16 for Shorts / Reels / TikTok
        cmd += ["-vf", "crop=ih*9/16:ih,scale=1080:1920"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)
    return dst
