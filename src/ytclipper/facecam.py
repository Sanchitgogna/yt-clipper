"""Find a streamer's webcam overlay in gameplay footage, automatically.

A facecam is a face that stays in the same spot while the game moves around it,
inside a rectangle whose edges do not move either. We look for both: a face
detected in the same place across the clip, then the still, straight edges
around it that mark the overlay.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

MODEL = Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"
ANALYSIS_W = 960


@dataclass
class Layout:
    """How to fill the vertical frame. All coordinates are fractions of the source frame."""
    cam: tuple[float, float, float, float] | None = None   # x, y, w, h of the facecam
    focus_x: float = 0.5                                   # where to centre a plain crop


def sample_frames(path: Path, start: float, length: float, count: int = 20) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    for i in range(count):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + length * (i + 0.5) / count) * 1000)
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            frames.append(cv2.resize(frame, (ANALYSIS_W, round(h * ANALYSIS_W / w) // 2 * 2)))
    cap.release()
    return frames


def find_faces(frames: list[np.ndarray]) -> list[list[tuple[float, float, float, float]]]:
    """Face boxes (x, y, w, h in pixels) for each frame."""
    import cv2

    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    h, w = frames[0].shape[:2]
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (w, h), score_threshold=0.75)
    out = []
    for f in frames:
        _, faces = det.detect(f)
        out.append([tuple(map(float, face[:4])) for face in (faces if faces is not None else [])])
    return out


def steady_face(faces: list[list[tuple]], frame_w: int) -> tuple[float, float, float, float] | None:
    """The face that sits in one place for most of the clip, if there is one."""
    best = None
    for seed in (f for frame in faces for f in frame):
        cx, cy = seed[0] + seed[2] / 2, seed[1] + seed[3] / 2
        near = [min((f for f in frame), default=None,
                    key=lambda f: abs(f[0] + f[2] / 2 - cx) + abs(f[1] + f[3] / 2 - cy))
                for frame in faces]
        near = [f for f in near if f and abs(f[0] + f[2] / 2 - cx) < 0.06 * frame_w
                and abs(f[1] + f[3] / 2 - cy) < 0.06 * frame_w]
        if len(near) >= 0.6 * len(faces) and (best is None or len(near) > len(best)):
            best = near
    if not best:
        return None
    return tuple(np.median(np.array(best), axis=0))


def _still_edges(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Edges present in (nearly) every frame: vertical ones, horizontal ones."""
    import cv2

    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    gx = np.abs(gray[:, :, 2:] - gray[:, :, :-2])
    gy = np.abs(gray[:, 2:, :] - gray[:, :-2, :])
    vert = np.pad(np.percentile(gx, 35, axis=0), ((0, 0), (1, 1)))
    horiz = np.pad(np.percentile(gy, 35, axis=0), ((1, 1), (0, 0)))
    return vert > 7, horiz > 7


def find_overlay(frames: list[np.ndarray], face: tuple) -> tuple[int, int, int, int] | None:
    """The still rectangle around a steady face: (x0, y0, x1, y1) in pixels, or None."""
    h, w = frames[0].shape[:2]
    vert, horiz = _still_edges(frames)
    fx, fy, fw, fh = face
    r0, r1 = int(max(0, fy)), int(min(h, fy + fh))          # rows the cam must span
    c0, c1 = int(max(0, fx)), int(min(w, fx + fw))          # columns it must span

    def candidates(strength: np.ndarray, start: int, stop: int, step: int, edge: int) -> list[int]:
        found, last = [], None
        for i in range(start, stop, step):
            if strength[i] > 0.7 and (last is None or abs(i - last) > 4):
                found.append(i)
            if strength[i] > 0.7:
                last = i
        return found[:4] + [edge]                           # the frame edge is always possible

    col = vert[r0:r1].mean(axis=0)
    row = horiz[:, c0:c1].mean(axis=1)
    reach_x, reach_y = int(4 * fw), int(3.5 * fh)
    lefts = candidates(col, c0 - 2, max(0, c0 - reach_x), -1, 0)
    rights = candidates(col, c1 + 2, min(w, c1 + reach_x), 1, w)
    tops = candidates(row, r0 - 2, max(0, r0 - reach_y), -1, 0)
    bottoms = candidates(row, r1 + 2, min(h, r1 + reach_y), 1, h)

    def support(mask: np.ndarray, a: int, b: int) -> float:
        return float(mask[a:b].mean()) if b > a else 0.0

    best, best_score = None, 0.0
    for x0, x1, y0, y1 in product(lefts, rights, tops, bottoms):
        cw, ch = x1 - x0, y1 - y0
        if not (0.95 <= cw / ch <= 2.4 and 0.1 <= fw / cw <= 0.75):
            continue                                        # not shaped like a webcam
        if cw > 0.7 * w and ch > 0.7 * h:
            continue                                        # that is the whole picture
        sides = []
        for pos, mask, a, b, edge in ((x0, vert[:, min(x0, w - 1)], y0, y1, 0),
                                      (x1, vert[:, min(x1, w - 1)], y0, y1, w),
                                      (y0, horiz[min(y0, h - 1)], x0, x1, 0),
                                      (y1, horiz[min(y1, h - 1)], x0, x1, h)):
            real = pos != edge
            sides.append((real, support(mask, a, b) if real else 0.7))
        # An overlay's border is a straight line along the whole side. Still shapes inside
        # the webcam (a shoulder, a shelf) are not, and the largest such frame is the overlay.
        real_sides = [s for real, s in sides if real]
        # (one side may be faint where the game happens to match the webcam's background)
        if len(real_sides) < 2 or max(real_sides) < 0.8 or min(real_sides) < 0.45:
            continue
        score = sum(s for _, s in sides) / 4 + 0.5 * (cw * ch) / (w * h)
        if score > best_score:
            best, best_score = (x0, y0, x1, y1), score
    return best


def detect(path: Path, start: float, length: float) -> Layout:
    """Decide the vertical layout for one clip."""
    try:
        frames = sample_frames(path, start, length)
        if len(frames) < 6:
            return Layout()
        h, w = frames[0].shape[:2]
        face = steady_face(find_faces(frames), w)
    except Exception:
        return Layout()                 # never let layout guessing stop a clip from being made
    if face is None:
        return Layout()
    if face[2] < 0.25 * w:
        box = find_overlay(frames, face)
        if box:
            x0, y0, x1, y1 = box
            return Layout(cam=(x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h))
    # no overlay, but someone holds still on screen: keep them inside the crop
    return Layout(focus_x=float(np.clip((face[0] + face[2] / 2) / w, 0, 1)))
