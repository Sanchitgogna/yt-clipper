"""Fetch videos (and their audience signals) from YouTube with yt-dlp."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

Progress = Callable[[str, "float | None"], None]


@dataclass
class Auth:
    cookies: str | None = None
    cookies_from_browser: str | None = None


@dataclass
class Source:
    path: Path                       # media to analyse (may be audio only)
    title: str = ""
    video_id: str = ""
    heatmap: list[dict] = field(default_factory=list)
    chat_path: Path | None = None
    captions_path: Path | None = None
    url: str | None = None

    def words(self) -> list[tuple[float, float, str]]:
        return load_captions(self.captions_path) if self.captions_path else []


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def _opts(workdir: Path, auth: Auth | None, **extra) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    opts = {
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "noprogress": True,
        # YouTube needs a JavaScript runtime to resolve formats; use whichever is installed
        "js_runtimes": {"deno": {}, "node": {}, "bun": {}},
        **extra,
    }
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:  # stated explicitly so yt-dlp also hands it to ffmpeg for range downloads
        opts["proxy"] = proxy
    if auth and auth.cookies:
        opts["cookiefile"] = auth.cookies
    if auth and auth.cookies_from_browser:
        opts["cookiesfrombrowser"] = (auth.cookies_from_browser,)
    return opts


def _hook(progress: Progress | None, stage: str):
    def hook(d: dict) -> None:
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            progress(stage, d["downloaded_bytes"] / total if total and d.get("downloaded_bytes") else None)
    return hook


def probe(url: str, workdir: Path, auth: Auth | None = None) -> dict:
    """Video metadata without downloading anything."""
    import yt_dlp

    with yt_dlp.YoutubeDL(_opts(workdir, auth)) as ydl:
        return ydl.extract_info(url, download=False)


def _downloaded(info: dict, workdir: Path) -> Path:
    for d in info.get("requested_downloads") or []:
        if d.get("filepath") and Path(d["filepath"]).exists():
            return Path(d["filepath"])
    found = [p for p in workdir.glob(f"{info['id']}.*")
             if p.suffix not in {".json", ".json3", ".part", ".ytdl"}]
    if not found:
        raise RuntimeError("download produced no media file")
    return found[0]


def _extras(url: str, workdir: Path, auth: Auth | None, chat: bool) -> None:
    """Captions and chat replay. Nice to have, so failures are ignored."""
    import yt_dlp

    langs = ["en", "en-orig", "en-US", "en-GB"] + (["live_chat"] if chat else [])
    try:
        with yt_dlp.YoutubeDL(_opts(workdir, auth, skip_download=True, writesubtitles=True,
                                    writeautomaticsub=True, subtitleslangs=langs,
                                    subtitlesformat="json3/best", ignoreerrors=True)) as ydl:
            ydl.extract_info(url, download=True)
    except Exception:
        pass


def _source(info: dict, path: Path, workdir: Path, url: str) -> Source:
    vid = info["id"]
    chat_path = workdir / f"{vid}.live_chat.json"
    captions = sorted(workdir.glob(f"{vid}.*.json3"))
    return Source(path=path, title=info.get("title") or vid, video_id=vid,
                  heatmap=info.get("heatmap") or [], url=url,
                  chat_path=chat_path if chat_path.exists() else None,
                  captions_path=captions[0] if captions else None)


def fetch(url: str, workdir: Path, max_height: int = 720, chat: bool = True,
          auth: Auth | None = None, progress: Progress | None = None,
          audio_only: bool = False) -> Source:
    """Download a video, or just its audio (enough to analyse, and far quicker)."""
    import yt_dlp

    if audio_only:
        fmt, extra = "ba/wv*+ba/w/b", {}
    else:
        fmt = f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b"
        extra = {"merge_output_format": "mp4"}
    stage = "Downloading audio" if audio_only else "Downloading video"
    with yt_dlp.YoutubeDL(_opts(workdir, auth, format=fmt, progress_hooks=[_hook(progress, stage)],
                                **extra)) as ydl:
        info = ydl.extract_info(url, download=True)
    path = _downloaded(info, workdir)
    if progress:
        progress("Getting captions and chat", None)
    _extras(url, workdir, auth, chat)
    return _source(info, path, workdir, url)


def fetch_sections(url: str, windows: list[tuple[int, int]], workdir: Path, max_height: int = 720,
                   auth: Auth | None = None, progress: Progress | None = None) -> list[Path]:
    """Download only the given (start, end) second ranges of a video."""
    import yt_dlp
    from yt_dlp.utils import download_range_func

    done = [0]

    def hook(d: dict) -> None:
        if d.get("status") == "finished":
            done[0] += 1
        if progress:
            progress("Downloading the best moments", min(1.0, done[0] / max(1, len(windows))))

    opts = _opts(workdir, auth, format=f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
                 merge_output_format="mp4", download_ranges=download_range_func(None, windows),
                 progress_hooks=[hook])
    opts["outtmpl"] = str(workdir / "%(id)s_sec%(section_start)d.%(ext)s")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    paths = []
    for a, _ in windows:
        found = [p for p in workdir.glob(f"{info['id']}_sec{a}.*") if p.suffix not in {".part", ".ytdl"}]
        if not found:
            raise RuntimeError(f"section at {a}s was not downloaded")
        paths.append(found[0])
    return paths


def heatmap_signal(heatmap: list[dict], n_seconds: int) -> np.ndarray:
    """YouTube's 'most replayed' graph, resampled to one value per second."""
    out = np.zeros(n_seconds)
    for seg in heatmap:
        a, b = int(seg["start_time"]), int(np.ceil(seg["end_time"]))
        out[max(0, a):min(n_seconds, b)] = seg["value"]
    return out


def chat_signal(chat_path: Path, n_seconds: int) -> np.ndarray:
    """Chat messages per second from a yt-dlp live_chat replay file."""
    out = np.zeros(n_seconds)
    with open(chat_path, encoding="utf-8") as f:
        for line in f:
            try:
                action = json.loads(line)["replayChatItemAction"]
                t = int(action["videoOffsetTimeMsec"]) // 1000
            except (ValueError, KeyError, TypeError):
                continue
            if 0 <= t < n_seconds:
                out[t] += 1
    return out


def load_captions(path: Path) -> list[tuple[float, float, str]]:
    """(start, end, word) from a YouTube json3 caption file."""
    try:
        events = json.loads(path.read_text(encoding="utf-8")).get("events", [])
    except (OSError, ValueError):
        return []
    pieces: list[tuple[float, float, str]] = []      # start, latest possible end, text
    for ev in events:
        t0 = ev.get("tStartMs", 0) / 1000
        ev_end = t0 + ev.get("dDurationMs", 0) / 1000
        segs = [(t0 + seg.get("tOffsetMs", 0) / 1000, seg.get("utf8", "").strip())
                for seg in ev.get("segs") or []]
        segs = [(t, text) for t, text in segs if text]
        for i, (t, text) in enumerate(segs):
            pieces.append((t, segs[i + 1][0] if i + 1 < len(segs) else max(ev_end, t), text))
    words = []
    for i, (t, limit, text) in enumerate(pieces):
        if i + 1 < len(pieces):
            limit = min(limit, pieces[i + 1][0]) if pieces[i + 1][0] > t else limit
        parts = [text] if text.startswith("[") else text.split()
        # manual captions come a line at a time: share the line's time between its words
        span = max(0.0, limit - t) if len(parts) > 1 else min(max(0.0, limit - t), 0.8)
        step = span / len(parts) if span else 0.3
        for j, part in enumerate(parts):
            words.append((t + j * step, t + (j + 1) * step, part))
    return words


_REACTION = re.compile(r"\[(laughter|laughs|applause|cheering|cheers|screams?|gasps?)\]", re.I)
_HYPE = {"wow", "whoa", "omg", "insane", "crazy", "unbelievable", "incredible", "holy",
         "clutch", "goal", "amazing", "impossible", "unreal", "wtf", "yooo", "nooo"}


def speech_signal(words: list[tuple[float, float, str]], n_seconds: int) -> np.ndarray:
    """Fast talking, hype words and audible reactions ([Laughter], [Applause]) per second."""
    out = np.zeros(n_seconds)
    for t, _, text in words:
        i = int(t)
        if not 0 <= i < n_seconds:
            continue
        if _REACTION.search(text):
            out[i] += 6
        elif text.startswith("["):   # [Music] and friends say nothing about excitement
            continue
        else:
            tokens = re.findall(r"[a-z']+", text.lower())
            out[i] += len(tokens) + 4 * sum(tok in _HYPE for tok in tokens) + 2 * text.count("!")
    return out


def text_between(words, start: float, end: float, limit: int = 400) -> str:
    text = " ".join(w for t, _, w in words if start <= t < end and not w.startswith("["))
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"
