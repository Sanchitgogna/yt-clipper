"""Animated captions drawn frame by frame.

The look and motion follow HyperFrames' "caption-highlight" component (Apache-2.0,
https://github.com/heygen-com/hyperframes): Montserrat ExtraBold, upper case, a few
words on screen at a time, and a red box that sweeps in behind the word being spoken.
Frames are drawn at output resolution, so the text stays sharp whatever the source.
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

Word = tuple[float, float, str]          # start, end, text (seconds, relative to the clip)

FONT = Path(__file__).parent / "assets" / "Montserrat-ExtraBold.ttf"
BASE_SIZE, MIN_SIZE, MAX_WIDTH = 84, 46, 900   # portrait fit, as in HyperFrames
PAD_X, PAD_TOP, PAD_BOTTOM, GAP, RADIUS = 12, 6, 8, 8, 10
SPACING = 0.02                                 # letter-spacing, in em
BOX_FROM, BOX_TO, GLOW = (255, 23, 69), (223, 18, 56), (229, 20, 58)
MARGIN = 48                                    # room around the line for shadows and glow
STRIP_H = BASE_SIZE + PAD_TOP + PAD_BOTTOM + 2 * MARGIN

_model = None
_lock = threading.Lock()  # one transcription at a time; the model already uses every core


def transcribe(media_path: Path, start: float, length: float) -> list[Word]:
    """Speech-to-text for one stretch of a file. Times are relative to `start`."""
    global _model
    from faster_whisper import WhisperModel

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
        return [(w.start, w.end, w.word.strip())
                for seg in segments for w in seg.words or [] if w.word.strip()]


@dataclass
class Group:
    start: float
    end: float
    words: list[Word]
    cut: bool = False     # the next phrase follows at once: switch to it without fading out


def group(words: list[Word], length: float, max_words: int = 3, max_chars: int = 18,
          max_gap: float = 0.35) -> list[Group]:
    """Short phrases, one on screen at a time: break at a sentence end, a pause, or the limit."""
    runs: list[list[Word]] = []
    for w in words:
        cur = runs[-1] if runs else None
        if cur:
            text = " ".join(x[2] for x in cur)
            if (len(cur) >= max_words or len(text) + 1 + len(w[2]) > max_chars
                    or w[0] - cur[-1][1] > max_gap or text[-1] in ".?!"):
                cur = None
        if cur is None:
            runs.append([w])
        else:
            cur.append(w)
    groups = []
    for i, run in enumerate(runs):
        nxt = runs[i + 1][0][0] if i + 1 < len(runs) else float("inf")
        end = min(run[-1][1] + 0.5, nxt, length)
        if end > run[0][0]:
            groups.append(Group(run[0][0], end, run, cut=nxt - run[-1][1] < 0.25))
    return groups


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), size)


def _text_width(text: str, size: int) -> float:
    f = _font(size)
    return sum(f.getlength(ch) for ch in text) + SPACING * size * len(text)


def _draw_text(draw: ImageDraw.ImageDraw, x: float, baseline: float, text: str, size: int, fill) -> None:
    f = _font(size)
    for ch in text:
        draw.text((x, baseline), ch, font=f, fill=fill, anchor="ls")
        x += f.getlength(ch) + SPACING * size


def _out(p: float) -> float:   # GSAP power2.out
    return 1 - (1 - min(1.0, max(0.0, p))) ** 2


def _in(p: float) -> float:    # GSAP power2.in
    return min(1.0, max(0.0, p)) ** 2


class _Line:
    """One group laid out and pre-drawn: a text layer, plus a box per word."""

    def __init__(self, g: Group, width: int):
        texts = [w[2].upper() for w in g.words]
        size = BASE_SIZE
        while size > MIN_SIZE and self._total(texts, size) > MAX_WIDTH:
            size -= 2
        self.group, self.width = g, width
        box_h = size + PAD_TOP + PAD_BOTTOM
        top = (STRIP_H - box_h) // 2
        x = (width - self._total(texts, size)) / 2
        baseline = top + PAD_TOP + size * 0.8585       # Montserrat's baseline at line-height 1

        text = Image.new("RGBA", (width, STRIP_H), (0, 0, 0, 0))
        soft, tight = Image.new("L", text.size, 0), Image.new("L", text.size, 0)
        self.boxes: list[tuple[int, int, Image.Image]] = []
        for t in texts:
            w = _text_width(t, size) + 2 * PAD_X
            _draw_text(ImageDraw.Draw(soft), x + PAD_X, baseline + 6, t, size, 255)
            _draw_text(ImageDraw.Draw(tight), x + PAD_X, baseline + 3, t, size, 255)
            self.boxes.append((round(x), top, self._box(round(w), box_h)))
            x += w + GAP
        black = Image.new("RGBA", text.size, (0, 0, 0, 255))
        # the soft shadow is HyperFrames'; the tight one keeps text readable over bright footage
        for mask, blur, alpha in ((soft, 9, 0.45), (tight, 3, 0.75)):
            m = mask.filter(ImageFilter.GaussianBlur(blur)).point(lambda v, a=alpha: int(v * a))
            text.paste(black, (0, 0), m)
        x = (width - self._total(texts, size)) / 2
        d = ImageDraw.Draw(text)
        for t in texts:
            _draw_text(d, x + PAD_X, baseline, t, size, (255, 255, 255, 255))
            x += _text_width(t, size) + 2 * PAD_X + GAP
        self.text = text

    @staticmethod
    def _total(texts: list[str], size: int) -> float:
        return sum(_text_width(t, size) + 2 * PAD_X for t in texts) + GAP * (len(texts) - 1)

    @staticmethod
    def _box(w: int, h: int) -> Image.Image:
        """Rounded gradient box with its glow, on a canvas MARGIN larger on every side."""
        canvas = Image.new("RGBA", (w + 2 * MARGIN, h + 2 * MARGIN), (0, 0, 0, 0))
        shape = Image.new("L", canvas.size, 0)
        ImageDraw.Draw(shape).rounded_rectangle((MARGIN, MARGIN + 12, MARGIN + w, MARGIN + 12 + h), RADIUS, fill=255)
        glow = shape.filter(ImageFilter.GaussianBlur(15)).point(lambda v: int(v * 0.32))
        canvas.paste(Image.new("RGBA", canvas.size, (*GLOW, 255)), (0, 0), glow)
        ramp = np.add.outer(np.arange(h), np.arange(w)) / max(1, w + h - 2)   # 135 degrees
        rgb = np.array(BOX_FROM) + ramp[..., None] * (np.array(BOX_TO) - np.array(BOX_FROM))
        fill = Image.fromarray(rgb.astype(np.uint8), "RGB")
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), RADIUS, fill=255)
        canvas.paste(fill, (MARGIN, MARGIN), mask)
        return canvas

    def draw(self, t: float) -> Image.Image:
        g = self.group
        frame = Image.new("RGBA", (self.width, STRIP_H), (0, 0, 0, 0))
        for (start, end, _), (x, y, box) in zip(g.words, self.boxes):
            fade_in = min(0.15, max(0.01, end - start))
            if start <= t < end:
                scale = opacity = _out((t - start) / fade_in)
            elif end <= t < end + 0.1:
                e = _in((t - end) / 0.1)
                scale, opacity = 1 + 0.02 * e, 1 - e
            else:
                continue
            w = max(1, round((box.width - 2 * MARGIN) * scale)) + 2 * MARGIN   # grows from the left
            layer = box.resize((w, box.height), Image.BILINEAR) if w != box.width else box
            if opacity < 1:
                layer = layer.copy()
                layer.putalpha(layer.getchannel("A").point(lambda v: int(v * opacity)))
            frame.alpha_composite(layer, (x - MARGIN, y - MARGIN))
        frame.alpha_composite(self.text)
        # in continuous speech phrases swap cleanly; fading every one makes the text flicker
        opacity = _out((t - g.start) / 0.08)
        if not g.cut:
            opacity = min(opacity, 1 - _in((t - (g.end - 0.1)) / 0.1))
        if opacity < 1:
            frame.putalpha(frame.getchannel("A").point(lambda v: int(v * opacity)))
        return frame


class Renderer:
    """Caption strip frames (RGBA, width x STRIP_H) for a clip."""

    def __init__(self, words: list[Word], length: float, fps: int = 30, width: int = 1080):
        self.fps, self.width, self.frames = fps, width, int(np.ceil(length * fps))
        self.lines = [_Line(g, width) for g in group(words, length)]
        self.blank = bytes(width * STRIP_H * 4)

    def __bool__(self) -> bool:
        return bool(self.lines)

    def frame(self, i: int) -> bytes:
        t = i / self.fps
        for line in self.lines:
            if line.group.start <= t < line.group.end:
                return line.draw(t).tobytes()
        return self.blank
