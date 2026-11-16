"""Watch a livestream and cut clips as exciting moments happen.

The stream is recorded into short segments. Each finished segment is scored
against a rolling baseline of the stream so far; when the loudness spikes well
above normal, the moment (plus lead-in and aftermath) is cut from the segments.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import media, scoring


class LiveDetector:
    """Online spike detector over a growing per-second loudness series."""

    def __init__(self, threshold: float = 4.0, pre_roll: float = 20, post_roll: float = 10,
                 cooldown: float = 30, warmup: float = 60, baseline: float = 600):
        self.threshold, self.pre_roll, self.post_roll = threshold, pre_roll, post_roll
        self.cooldown, self.warmup, self.baseline = cooldown, warmup, baseline
        self.energy = np.zeros(0)
        self.cursor = 0  # first second not yet judged

    def update(self, new_energy: np.ndarray) -> list[scoring.Clip]:
        self.energy = np.concatenate([self.energy, new_energy])
        clips = []
        while True:
            lo = max(self.cursor, int(self.warmup))
            hi = len(self.energy) - int(self.post_roll)  # need the aftermath recorded
            if hi <= lo:
                break
            smoothed = scoring.smooth(self.energy, 3)
            base = smoothed[max(0, lo - int(self.baseline)):lo]
            # min_scale: a near-silent stream must not make tiny noises look huge
            z = scoring.robust_z(smoothed[lo:hi], min_scale=1.5, baseline=base)
            if z.max() < self.threshold:
                self.cursor = hi
                break
            # take the whole burst, not just its first second
            first = int(np.argmax(z >= self.threshold))
            last = first
            while last + 1 < len(z) and z[last + 1] >= self.threshold * 0.5:
                last += 1
            if lo + last + 1 >= hi and last - first < 60:
                break  # burst still going; wait for more of the stream
            peak = lo + first + int(np.argmax(z[first:last + 1]))
            start = max(0.0, lo + first - self.pre_roll)
            end = lo + last + 1 + self.post_roll
            clips.append(scoring.Clip(start, float(min(end, len(self.energy))),
                                      float(peak), float(z[peak - lo])))
            self.cursor = int(lo + last + 1 + self.cooldown)
        return clips


@dataclass
class Segment:
    path: Path
    start: float
    length: float


def cut_from_segments(segments: list[Segment], clip: scoring.Clip, dst: Path,
                      vertical: bool = False) -> Path | None:
    used = [s for s in segments
            if s.start < clip.end and s.start + s.length > clip.start and s.path.exists()]
    if not used:
        return None
    listing = dst.with_suffix(".txt")
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text("".join(f"file '{s.path.resolve()}'\n" for s in used))
    try:
        offset = max(0.0, clip.start - used[0].start)
        return media.cut(listing, dst, offset, clip.end - max(clip.start, used[0].start),
                         vertical=vertical, concat_list=True)
    finally:
        listing.unlink(missing_ok=True)


def start_recorder(url: str, seg_dir: Path, segment_time: int = 15) -> list[subprocess.Popen]:
    """Record a stream into seg_dir as numbered .ts segments."""
    seg_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(seg_dir / "seg_%06d.ts")
    seg_args = ["-c", "copy", "-f", "segment", "-segment_time", str(segment_time),
                "-reset_timestamps", "1", pattern]
    if url.startswith(("http://", "https://")):
        dl = subprocess.Popen(["yt-dlp", "-q", "--no-warnings", "--js-runtimes", "node",
                               "--js-runtimes", "deno", "-f", "b[height<=720]/b",
                               "-o", "-", url], stdout=subprocess.PIPE)
        ff = subprocess.Popen(["ffmpeg", "-v", "error", "-i", "pipe:0", *seg_args], stdin=dl.stdout)
        assert dl.stdout is not None
        dl.stdout.close()
        return [ff, dl]
    # a local file played back in real time: handy for trying live mode offline
    return [subprocess.Popen(["ffmpeg", "-v", "error", "-re", "-i", url, *seg_args])]


def run(url: str, out_dir: Path, work_dir: Path, detector: LiveDetector | None = None,
        vertical: bool = False, keep: float = 900, poll: float = 2.0, log=print) -> list[Path]:
    """Record `url` until it ends (or Ctrl-C), writing clips to out_dir as they happen."""
    detector = detector or LiveDetector()
    seg_dir = work_dir / "segments"
    for old in seg_dir.glob("seg_*.ts") if seg_dir.exists() else []:
        old.unlink()
    procs = start_recorder(url, seg_dir)
    segments: list[Segment] = []
    done: set[Path] = set()
    written: list[Path] = []
    clock = 0.0
    try:
        while True:
            running = procs[0].poll() is None
            files = sorted(seg_dir.glob("seg_*.ts"))
            ready = files if not running else files[:-1]  # last one is still being written
            for path in ready:
                if path in done:
                    continue
                done.add(path)
                energy = media.audio_energy(path)
                seg = Segment(path, clock, float(len(energy)))
                segments.append(seg)
                clock += seg.length
                for clip in detector.update(energy):
                    dst = out_dir / f"live_{int(clip.start):06d}s.mp4"
                    if cut_from_segments(segments, clip, dst, vertical):
                        written.append(dst)
                        log(f"clip  {fmt(clip.start)}-{fmt(clip.end)}  score {clip.score:.1f}  -> {dst}")
                while segments and segments[0].start + segments[0].length < clock - keep:
                    segments.pop(0).path.unlink(missing_ok=True)
            if not running:
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        log("stopping...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    return written


def fmt(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
