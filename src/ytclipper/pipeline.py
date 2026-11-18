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

from . import captions, facecam, judge, media, render, scoring, source
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
    smart: bool = True         # read what is said in the candidates before choosing
    claude: bool = True        # let Claude choose and title them, when an API key is set
    notes: list[str] = field(default_factory=list)   # things the user should know about this run
    login_failed: bool = False                         # a browser login was chosen but unreadable
    auth: Auth = field(default_factory=Auth)


@dataclass
class Result:
    start: float
    end: float
    score: float
    path: Path | None = None
    text: str = ""             # what is said in the clip, when captions exist
    facecam: bool = False      # a webcam overlay was found and placed on top
    words: list | None = None  # (start, end, word) in video time, once listened to
    title: str = ""            # Claude's title for the clip
    rating: int = 0            # Claude's 1-10 rating
    reason: str = ""           # ... and why


def _say(progress: Progress | None, stage: str, frac: float | None = None) -> None:
    if progress:
        progress(stage, frac)


def analyze(src: source.Source, scenes: bool = True,
            voice: bool = False) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
    elif voice and len(audio):
        # no captions to read: at least find where people talk, so that spoken moments
        # compete with loud ones (the candidates are transcribed later)
        active = captions.voice_activity(src.path)
        if active.sum() > 0:
            signals["voice"] = active
    if not signals:
        return np.zeros(n), signals
    return scoring.combine(signals), signals


PAD_LISTEN = 2.5   # seconds heard beyond each edge, so a clip can grow to a sentence boundary


def _listen(src: source.Source, r: Result) -> None:
    """Fill in r.words: what is said in and just around the candidate."""
    a, b = max(0.0, r.start - PAD_LISTEN), r.end + PAD_LISTEN
    words = [w for w in src.words() if a <= w[0] < b and not w[2].startswith("[")]
    if not words:
        words = [(a + x, a + y, w) for x, y, w in captions.transcribe(src.path, a, b - a)]
    r.words = words


def _snap_to_sentences(r: Result, s: Settings) -> None:
    """Start on the first word of a sentence and end on the last word of one."""
    words = r.words or []
    starts = [w[0] for i, w in enumerate(words)
              if i == 0 or words[i - 1][2][-1] in ".?!" or w[0] - words[i - 1][1] > 0.7]
    ends = [w[1] for i, w in enumerate(words)
            if i == len(words) - 1 or w[2][-1] in ".?!" or words[i + 1][0] - w[1] > 0.7]
    start = min(starts, key=lambda t: abs(t - r.start), default=r.start) - 0.15
    end = min(ends, key=lambda t: abs(t - r.end), default=r.end) + 0.3
    if abs(start - r.start) <= PAD_LISTEN and abs(end - r.end) <= PAD_LISTEN \
            and s.min_len <= end - start <= s.max_len + PAD_LISTEN:
        r.start, r.end = max(0.0, start), end


def _interest(r: Result) -> float:
    """Rough measure of how much is going on in the words, per second."""
    points = 0.0
    for _, _, w in r.words or []:
        token = w.lower().strip(".,!?\"'")
        points += 1 + 4 * (token in source._HYPE) + 2 * w.count("!") + w.count("?")
    return points / max(1.0, r.end - r.start)


def _rank(src: source.Source, pool: list[Result], s: Settings) -> list[Result]:
    """Order candidates best first: Claude's judgement if available, else heuristics."""
    if s.claude and judge.source():
        try:
            verdicts = judge.rate(src.title, [
                {"index": i, "start": r.start, "end": r.end, "transcript": r.text}
                for i, r in enumerate(pool)])
            for i, r in enumerate(pool):
                if i in verdicts:
                    v = verdicts[i]
                    r.title, r.rating, r.reason = v.title, v.rating, v.reason
            top = max(r.score for r in pool) or 1.0
            for r in pool:   # the rating decides; the signal score only breaks ties
                r.score = (r.rating or 5) + 0.5 * r.score / top
            keep = [r for r in pool if r.rating != 0 and r.rating > 2] or pool
            return sorted(keep, key=lambda r: -r.score)
        except Exception as e:
            s.notes.append(f"Claude could not rate the clips ({judge.explain(e)}), "
                           "so they were ranked without it.")
    interest = np.array([_interest(r) for r in pool])
    if interest.std() > 1e-9:
        z = np.clip((interest - interest.mean()) / interest.std(), -1, 2)
        for r, boost in zip(pool, z):
            r.score *= 1 + 0.25 * float(boost)
    return sorted(pool, key=lambda r: -r.score)


def find(src: source.Source, s: Settings, progress: Progress | None = None) -> list[Result]:
    """The best moments of a video, in the order they happen."""
    score, signals = analyze(src, scenes=s.scenes, voice=s.smart)
    # look at more candidates than needed, so the ones chosen are picked for what is in them
    pool_size = min(12, s.count * 2) if s.smart else s.count
    clips = scoring.pick_clips(score, pool_size, s.min_len, s.max_len)
    if "audio" in signals:
        clips = scoring.snap_to_quiet(clips, signals["audio"], s.min_len, s.max_len)
    pool = [Result(c.start, c.end, c.score) for c in clips]
    words = src.words()
    if not s.smart:
        for r in pool:
            r.text = source.text_between(words, r.start, r.end)
        return pool
    for i, r in enumerate(pool):
        _say(progress, "Listening to the best moments", i / len(pool))
        _listen(src, r)
        _snap_to_sentences(r, s)
        r.text = source.text_between(r.words, r.start, r.end, limit=1500)
    _say(progress, "Choosing the best clips")
    return sorted(_rank(src, pool, s)[:s.count], key=lambda r: r.start)


def slug(text: str, limit: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:limit] or "clip"


def _names(title: str, results: list[Result], out_dir: Path) -> list[Path]:
    ranked = sorted(results, key=lambda r: -r.score)
    return [out_dir / (f"{slug(title, 30)}_top{ranked.index(r) + 1:02d}_"
                       + (slug(r.title) if r.title else f"{int(r.start):05d}s") + ".mp4")
            for r in results]


def _parallel(jobs: list, progress: Progress | None, stage: str) -> None:
    done = 0
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as pool:
        for _ in pool.map(lambda job: job(), jobs):
            done += 1
            _say(progress, stage, done / len(jobs))


def _words(src: source.Source, part: Path, start: float, r: Result) -> list[captions.Word]:
    """The clip's words: YouTube's captions if it has them, else speech-to-text."""
    known = r.words if r.words is not None else src.words()
    words = [(a - r.start, min(b, r.end) - r.start, w) for a, b, w in known
             if r.start <= a < r.end and not w.startswith("[")]
    if not words and r.words is None:
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
    return src, find(src, s, progress)


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
    results = find(src, s, progress)
    return src, export(src, results, out_dir, s, progress) if results else results


def cleanup(work_dir: Path) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
