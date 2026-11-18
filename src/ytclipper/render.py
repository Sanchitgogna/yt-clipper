"""Render one vertical 1080x1920 clip: layout, captions, encode."""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import captions, media
from .facecam import Layout

W, H, FPS = 1080, 1920, 30


def _even(v: float) -> int:
    return int(round(v / 2)) * 2


def _graph(src_w: int, src_h: int, layout: Layout) -> tuple[str, int]:
    """ffmpeg filter producing [v] at 1080x1920, and the y where captions go."""
    scale = "flags=lanczos"
    if layout.cam:
        x, y, w, h = layout.cam
        cw, ch = _even(w * src_w), _even(h * src_h)
        cx, cy = _even(x * src_w), _even(y * src_h)
        top = _even(min(760, max(480, W * ch / cw)))         # facecam panel height
        bottom = H - top
        gw = _even(min(src_w, src_h * W / bottom))           # game crop, same shape as its panel
        gh = _even(min(src_h, gw * bottom / W))
        gx = (src_w - gw) / 2
        # slide the game crop away from the facecam where there is room, so it is not shown twice
        cam_mid = cx + cw / 2
        if cam_mid < src_w / 2:
            gx = min(src_w - gw, max(gx, min(cx + cw, gx + 0.15 * src_w)))
        else:
            gx = max(0, min(gx, max(cx - gw, gx - 0.15 * src_w)))
        return (f"[0:v]fps={FPS},split[a][b];"
                f"[a]crop={cw}:{ch}:{cx}:{cy},scale={W}:{top}:force_original_aspect_ratio=increase:{scale},"
                f"crop={W}:{top}[cam];"
                f"[b]crop={gw}:{gh}:{_even(gx)}:{_even((src_h - gh) / 2)},scale={W}:{bottom}:{scale}[game];"
                f"[cam][game]vstack,setsar=1[v]"), top + 16
    if src_w / src_h >= W / H:                               # wider than 9:16: crop the sides
        cw = _even(src_h * W / H)
        cx = _even(min(src_w - cw, max(0, layout.focus_x * src_w - cw / 2)))
        crop = f"crop={cw}:{src_h}:{cx}:0"
    else:                                                    # taller: crop top and bottom
        ch = _even(src_w * H / W)
        crop = f"crop={src_w}:{ch}:0:{_even((src_h - ch) / 2)}"
    return f"[0:v]fps={FPS},{crop},scale={W}:{H}:{scale},setsar=1[v]", H - 700


def render(src: Path, dst: Path, start: float, length: float, layout: Layout | None = None,
           words: list[captions.Word] | None = None) -> Path:
    """Cut [start, start+length) of src into a vertical clip at dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_w, src_h = media.size(src)
    graph, caption_y = _graph(src_w, src_h, layout or Layout())
    caps = captions.Renderer(words, length, FPS, W) if words else None
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src)]
    if caps:
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{captions.STRIP_H}",
                "-r", str(FPS), "-i", "pipe:0"]
        graph += f";[v][1:v]overlay=0:{caption_y}:eof_action=pass[out]"
    cmd += ["-filter_complex", graph, "-map", "[out]" if caps else "[v]", "-map", "0:a:0?",
            "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
            "-t", f"{length:.3f}", str(dst)]
    # stdout is never ffmpeg's: under the MCP server it carries the protocol
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if caps else None, stdout=subprocess.DEVNULL)
    if caps:
        try:
            for i in range(caps.frames):
                proc.stdin.write(caps.frame(i))
        except BrokenPipeError:
            pass                                             # ffmpeg stopped early; its exit code says why
        finally:
            proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg could not render {dst.name}")
    return dst
