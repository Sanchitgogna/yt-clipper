"""Fetch a video (and its audience signals) from YouTube with yt-dlp."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Source:
    path: Path
    title: str = ""
    video_id: str = ""
    heatmap: list[dict] = field(default_factory=list)
    chat_path: Path | None = None


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def fetch(url: str, workdir: Path, max_height: int = 720, chat: bool = True) -> Source:
    import yt_dlp

    workdir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if chat:
        opts |= {"writesubtitles": True, "subtitleslangs": ["live_chat"]}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    if not info:
        raise RuntimeError(f"could not download {url}")
    vid = info["id"]
    path = workdir / f"{vid}.mp4"
    if not path.exists():
        found = [p for p in workdir.glob(f"{vid}.*") if p.suffix not in {".json", ".part"}]
        if not found:
            raise RuntimeError(f"download produced no media file for {url}")
        path = found[0]
    chat_path = workdir / f"{vid}.live_chat.json"
    return Source(
        path=path,
        title=info.get("title") or vid,
        video_id=vid,
        heatmap=info.get("heatmap") or [],
        chat_path=chat_path if chat_path.exists() else None,
    )


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
