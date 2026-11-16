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

from . import captions, facecam, media, render, scoring, source
from .source import Auth, Progress

PAD = 3  # extra seconds fetched around a section so the precise cut has room


class SectionsUnavailable(RuntimeError):
    """The site would not serve parts of the video; the whole file is needed."""


@dataclass
class Settings:
    count: int = 5
    min_len: float = 15
    max_len: float = 60
    max_height: int = 1080     # a vertical crop keeps about a third of the width: start sharp
    scenes: bool = False       # scene-cut detection needs the whole video: slow
    chat: bool = True
    fast: bool = True          # analyse audio first, then download only the clips
    captions: bool = True      # animated captions of what is said
    facecam: bool = True       # put a streamer's webcam on top when there is one
    auth: Auth = field(default_factory=Auth)


@dataclass
class Result:
    start: float
    end: float
    score: float
    path: Path | None = None
    text: str = ""             # what is said in the clip, when captions exist
    facecam: bool = False      # a webcam overlay was found and placed on top


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


def _words(src: source.Source, part: Path, start: float, r: Result) -> list[captions.Word]:
    """The clip's words: YouTube's captions if it has them, else speech-to-text."""
    words = [(a - r.start, b - r.start, w) for a, b, w in src.words()
             if r.start <= a < r.end and not w.startswith("[")]
    if not words:
        words = captions.transcribe(part, start, r.end - r.start)
        r.text = r.text or " ".join(w for *_, w in words)
    return words


def _cut(part: Path, start: float, src: source.Source, r: Result, s: Settings) -> None:
    """Render one clip. `part` holds the footage, with the clip beginning at `start`."""
    length = r.end - r.start
    layout = facecam.detect(part, start, length) if s.facecam else facecam.Layout()
    r.facecam = layout.cam is not None
    words = _words(src, part, start, r) if s.captions else None
    render.render(part, r.path, start, length, layout, words)


def export(src: source.Source, results: list[Result], out_dir: Path, s: Settings,
           progress: Progress | None = None) -> list[Result]:
    """Cut clips out of a fully downloaded (or local) video."""
    for r, dst in zip(results, _names(src.title or src.path.stem, results, out_dir)):
        r.path = dst
    _say(progress, "Making clips", 0)
    _parallel([lambda r=r: _cut(src.path, r.start, src, r, s) for r in results],
              progress, "Making clips")
    return results


def export_sections(url: str, src: source.Source, results: list[Result], out_dir: Path,
                    work_dir: Path, s: Settings, progress: Progress | None = None) -> list[Result]:
    """Download only the clip ranges of a video and trim them precisely.
    `src` is the analysed audio: it supplies the title and the words for captions."""
    windows = [(max(0, math.floor(r.start) - PAD), math.ceil(r.end) + PAD) for r in results]
    _say(progress, "Downloading the best moments", 0)
    try:
        parts = source.fetch_sections(url, windows, work_dir, s.max_height, s.auth, progress)
    except Exception as e:
        raise SectionsUnavailable(str(e)) from e

    def trim(r: Result, part: Path, window: tuple[int, int]) -> None:
        # A stream-copied section starts on an earlier keyframe, so its head is
        # unpredictable; its tail is not. Measure the clip back from the end.
        start = max(0.0, media.duration(part) - (window[1] - r.end) - (r.end - r.start))
        _cut(part, start, src, r, s)

    for r, dst in zip(results, _names(src.title, results, out_dir)):
        r.path = dst
    stage = "Making clips"
    _say(progress, stage, 0)
    _parallel([lambda a=a: trim(*a) for a in zip(results, parts, windows)], progress, stage)
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
            return src, export_sections(url, src, results, out_dir, work_dir, s, progress)
        except SectionsUnavailable:
            # some sites cannot serve ranges; fall back to the whole video
            src.path = source.fetch(url, work_dir / "full", s.max_height, False, s.auth, progress).path
    return src, export(src, results, out_dir, s, progress)


def clip_file(path: Path, out_dir: Path, s: Settings,
              progress: Progress | None = None) -> tuple[source.Source, list[Result]]:
    src = source.Source(path=path, title=path.stem)
    _say(progress, "Finding the best moments")
    results = find(src, s)
    return src, export(src, results, out_dir, s, progress) if results else results


def cleanup(work_dir: Path) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
