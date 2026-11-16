"""Analyse a finished video and cut its best moments."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import media, scoring, source


def analyze(src: source.Source, scenes: bool = True) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (highlight score per second, the raw signals it was built from)."""
    signals: dict[str, np.ndarray] = {"audio": media.audio_energy(src.path)}
    n = max(len(signals["audio"]), int(media.duration(src.path)))
    if scenes:
        signals["scenes"] = media.scene_changes(src.path, n)
    if src.heatmap:
        signals["heatmap"] = source.heatmap_signal(src.heatmap, n)
    if src.chat_path:
        chat = source.chat_signal(src.chat_path, n)
        if chat.sum() > 0:
            signals["chat"] = chat
    return scoring.combine(signals), signals


def slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:limit] or "clip"


def export(src: source.Source, clips: list[scoring.Clip], out_dir: Path,
           vertical: bool = False) -> list[Path]:
    name = slug(src.title or src.path.stem)
    paths = []
    ranked = sorted(clips, key=lambda c: -c.score)
    for clip in clips:
        rank = ranked.index(clip) + 1
        dst = out_dir / f"{name}_top{rank:02d}_{int(clip.start):05d}s.mp4"
        paths.append(media.cut(src.path, dst, clip.start, clip.length, vertical=vertical))
    return paths
