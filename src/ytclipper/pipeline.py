"""Find the best moments of a finished video and cut them."""

from __future__ import annotations

import math
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import media, scoring, source
from .source import Auth, Progress

PAD = 3  # extra seconds fetched around a section so the precise cut has room


@dataclass
class Settings:
    count: int = 5
    min_len: float = 15
    max_len: float = 60
    vertical: bool = False
    max_height: int = 720
    scenes: bool = False       # scene-cut detection needs the whole video: slow
    chat: bool = True
    fast: bool = True          # analyse audio first, then download only the clips
    auth: Auth = field(default_factory=Auth)


@dataclass
class Result:
    start: float
    end: float
    score: float
    path: Path | None = None
    text: str = ""             # what is said in the clip, when captions exist


def _say(progress: Progress | None, stage: str, frac: float | None = None) -> None:
    if progress:
        progress(stage, frac)


def analyze(src: source.Source, scenes: bool = True) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (highlight score per second, the raw signals it was built from)."""
    audio = media.audio_energy(src.path)
    n = max(len(audio), int(media.duration(src.path)))
    signals: dict[str, np.ndarray] = {"audio": audio} if len(audio) else {}  # silent video
    if scenes:
        signals["scenes"] = media.scene_changes(src.path, n)
    if src.heatmap:
        signals["heatmap"] = source.heatmap_signal(src.heatmap, n)
    if src.chat_path:
        chat = source.chat_signal(src.chat_path, n)
        if chat.sum() > 0:
            signals["chat"] = chat
    words = src.words()
    if words:
        signals["speech"] = source.speech_signal(words, n)
    if not signals:
        return np.zeros(n), signals
    return scoring.combine(signals), signals


def find(src: source.Source, s: Settings) -> list[Result]:
    score, signals = analyze(src, scenes=s.scenes)
    clips = scoring.pick_clips(score, s.count, s.min_len, s.max_len)
    if "audio" in signals:
        clips = scoring.snap_to_quiet(clips, signals["audio"], s.min_len, s.max_len)
    words = src.words()
    return [Result(c.start, c.end, c.score, text=source.text_between(words, c.start, c.end))
            for c in clips]


def slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:limit] or "clip"


def _names(title: str, results: list[Result], out_dir: Path) -> list[Path]:
    ranked = sorted(results, key=lambda r: -r.score)
    return [out_dir / f"{slug(title)}_top{ranked.index(r) + 1:02d}_{int(r.start):05d}s.mp4"
            for r in results]


def _parallel(jobs: list, progress: Progress | None, stage: str) -> None:
    done = 0
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as pool:
        for _ in pool.map(lambda job: job(), jobs):
            done += 1
            _say(progress, stage, done / len(jobs))


def export(src: source.Source, results: list[Result], out_dir: Path, vertical: bool = False,
           progress: Progress | None = None) -> list[Result]:
    """Cut clips out of a fully downloaded (or local) video."""
    for r, dst in zip(results, _names(src.title or src.path.stem, results, out_dir)):
        r.path = dst
    _say(progress, "Cutting clips", 0)
    _parallel([lambda r=r: media.cut(src.path, r.path, r.start, r.end - r.start, vertical=vertical)
               for r in results], progress, "Cutting clips")
    return results


def export_sections(url: str, title: str, results: list[Result], out_dir: Path, work_dir: Path,
                    s: Settings, progress: Progress | None = None) -> list[Result]:
    """Download only the clip ranges of a video and trim them precisely."""
    windows = [(max(0, math.floor(r.start) - PAD), math.ceil(r.end) + PAD) for r in results]
    _say(progress, "Downloading the best moments", 0)
    parts = source.fetch_sections(url, windows, work_dir, s.max_height, s.auth, progress)

    def trim(r: Result, part: Path, window: tuple[int, int]) -> None:
        # A stream-copied section starts on an earlier keyframe, so its head is
        # unpredictable; its tail is not. Measure the clip back from the end.
        length = r.end - r.start
        start = max(0.0, media.duration(part) - (window[1] - r.end) - length)
        media.cut(part, r.path, start, length, vertical=s.vertical)

    for r, dst in zip(results, _names(title, results, out_dir)):
        r.path = dst
    _say(progress, "Cutting clips", 0)
    _parallel([lambda a=a: trim(*a) for a in zip(results, parts, windows)], progress, "Cutting clips")
    return results


def find_url(url: str, work_dir: Path, s: Settings,
             progress: Progress | None = None) -> tuple[source.Source, list[Result]]:
    src = source.fetch(url, work_dir, s.max_height, s.chat, s.auth, progress,
                       audio_only=s.fast and not s.scenes)
    _say(progress, "Finding the best moments")
    return src, find(src, s)


def clip_url(url: str, out_dir: Path, work_dir: Path, s: Settings,
             progress: Progress | None = None) -> tuple[source.Source, list[Result]]:
    """The whole job for a video URL: analyse, pick, download, cut."""
    src, results = find_url(url, work_dir, s, progress)
    if not results:
        return src, results
    if s.fast and not s.scenes:
        try:
            return src, export_sections(url, src.title, results, out_dir, work_dir, s, progress)
        except Exception:
            # some sites cannot serve ranges; fall back to the whole video
            src.path = source.fetch(url, work_dir / "full", s.max_height, False, s.auth, progress).path
    return src, export(src, results, out_dir, s.vertical, progress)


def clip_file(path: Path, out_dir: Path, s: Settings,
              progress: Progress | None = None) -> tuple[source.Source, list[Result]]:
    src = source.Source(path=path, title=path.stem)
    _say(progress, "Finding the best moments")
    results = find(src, s)
    return src, export(src, results, out_dir, s.vertical, progress) if results else results


def cleanup(work_dir: Path) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
