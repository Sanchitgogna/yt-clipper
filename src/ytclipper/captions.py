"""Burned-in captions: get the words of a clip and render them as subtitles."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import numpy as np

Word = tuple[float, str]

_model = None
_lock = threading.Lock()  # one transcription at a time; the model already uses every core


def transcribe(media_path: Path, start: float, length: float) -> list[Word]:
    """Speech-to-text for one stretch of a file. Times are relative to `start`."""
    global _model
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError("captions for videos without YouTube captions need speech-to-text: "
                           "pip install 'ytclipper[captions]'") from e
    # decode with ffmpeg ourselves: 16 kHz mono float samples are what the model wants
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(media_path),
                          "-t", f"{length:.3f}", "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                          "-f", "s16le", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if not len(audio):
        return []
    with _lock:
        if _model is None:
            _model = WhisperModel(os.environ.get("YTCLIPPER_WHISPER_MODEL", "small"),
                                  device="cpu", compute_type="int8")
        segments, _ = _model.transcribe(audio, word_timestamps=True, vad_filter=True)
        return [(w.start, w.word.strip()) for seg in segments for w in seg.words or [] if w.word.strip()]


def group(words: list[Word], length: float, max_chars: int = 26, max_gap: float = 0.8,
          max_show: float = 2.5) -> list[tuple[float, float, str]]:
    """Split words into short on-screen phrases: (start, end, text)."""
    lines: list[tuple[float, float, str]] = []
    cur: list[Word] = []

    def flush(next_start: float) -> None:
        if cur:
            start = cur[0][0]
            end = min(next_start, cur[-1][0] + max_show, length)
            if end > start:
                lines.append((start, end, " ".join(w for _, w in cur)))
            cur.clear()

    for t, w in words:
        if cur:
            text = " ".join(x for _, x in cur)
            if (len(text) + 1 + len(w) > max_chars or t - cur[-1][0] > max_gap
                    or text[-1] in ".?!"):
                flush(t)
        cur.append((t, w))
    flush(length)
    return lines


def _ts(t: float) -> str:
    cs = max(0, round(t * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def write_ass(words: list[Word], length: float, dst: Path, vertical: bool) -> Path | None:
    """Write a styled subtitle file for a clip; None when nothing is said."""
    lines = group(words, length, max_chars=16 if vertical else 30)
    if not lines:
        return None
    w, h, size, margin = (1080, 1920, 96, 460) if vertical else (1280, 720, 56, 44)
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Cap,Arial,{size},&H00FFFFFF,&H00000000,&H80000000,-1,1,{size / 14:.1f},{size / 40:.1f},2,60,60,{margin}

[Events]
Format: Layer, Start, End, Style, Text
"""
    body = "".join(f"Dialogue: 0,{_ts(a)},{_ts(b)},Cap,{text.replace('{', '(').replace('}', ')')}\n"
                   for a, b, text in lines)
    dst.write_text(head + body, encoding="utf-8")
    return dst
